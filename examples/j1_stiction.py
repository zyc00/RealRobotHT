"""Calibrate J1's static friction (one direction) and feel it compensated.

Protocol (J2-J6 firmware-PD pinned throughout, one TorqueSession):
  1. Station J1 on a grid from --start to --end (right -> left by default).
  2. At each station: settle pinned, then free J1 and ramp a feed-forward
     torque slowly (RATE N.m/s). Breakaway = the torque at which J1 has moved
     BREAK_DEG; also record the slip JUMP in the next 0.3 s (the spike you
     feel when hand-guiding). J1 is a vertical axis: gravity torque ~0, so
     the ramp fights friction only.
  3. Print the profile (stiction varies with gear position -> max + table),
     save data/j1_stiction.npz.
  4. Compensation feel phase: J1 freed with tau1 = alpha * profile(q1) in the
     sweep direction (a/A adjusts alpha, q quits). Compensation is zeroed
     within GUARD deg of the sweep end so it cannot drive into the limit.
     alpha < 1: feed slightly less than breakaway or the joint self-drives
     wherever local stiction dips below the fed value.

THE ARM MOVES BY ITSELF (J1 sweeps the full grid): CLEAR THE WORKSPACE.

    python examples/j1_stiction.py                       # -50 -> +50 deg, 5 deg grid
    python examples/j1_stiction.py --start 40 --end -40  # the other direction
    python examples/j1_stiction.py --step 10 --cap 1.2
"""
import argparse
import os
import select
import sys
import termios
import time
import tty

_PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
try:
    from piperx_teleop import MitCommand, PiperModel, TorqueSession, require_patched_sdk
    require_patched_sdk()
except (RuntimeError, ImportError):
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

import numpy as np

RAD = np.pi / 180.0
BASE = np.radians([0.0, 45.0, -70.0, 0.0, 10.0, 0.0])   # friction_id pose, J4 near 0
PIN_KP, PIN_KD = 3.0, 0.8
RATE = 0.15          # N.m/s torque ramp - slow enough to resolve breakaway
BREAK_DEG = 0.3      # motion that counts as broken away
JUMP_S = 0.3         # window to measure the post-breakaway slip
GUARD_DEG = 6.0      # no compensation this close to the sweep end

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=float, default=-50.0, help="deg (right)")
ap.add_argument("--end", type=float, default=50.0, help="deg (left)")
ap.add_argument("--step", type=float, default=5.0, help="deg between stations")
ap.add_argument("--cap", type=float, default=1.5, help="N.m ramp ceiling")
ap.add_argument("--alpha", type=float, default=0.8, help="compensation fraction")
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

DIR = 1.0 if a.end > a.start else -1.0
stations = np.arange(a.start, a.end + 1e-9 * DIR, a.step * DIR) * RAD
mdl = PiperModel()


class J1Law:
    """pin: all joints PD-held at q_ref.  ramp: J1 free + slow torque ramp.
    comp: J1 free + alpha * profile(q1) feed-forward."""

    def __init__(self):
        self.mode = "pin"
        self.q_ref = None
        self.u_t0 = None
        self.u = 0.0
        self.alpha = a.alpha
        self.profile = None          # (q1_grid, tau_break) after calibration

    def comp_torque(self, q1):
        lo, hi = min(a.start, a.end) + GUARD_DEG, max(a.start, a.end) - GUARD_DEG
        if not (lo * RAD < q1 < hi * RAD):
            return 0.0
        return self.alpha * np.interp(q1, self.profile[0], self.profile[1]) * DIR

    def __call__(self, s):
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        tau = mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        if self.mode == "ramp":
            if self.u_t0 is None:
                self.u_t0 = s.t
            self.u = min(RATE * (s.t - self.u_t0), a.cap)
            kp[0] = kd[0] = 0.0
            tau[0] += self.u * DIR
        elif self.mode == "comp":
            kp[0] = kd[0] = 0.0
            tau[0] += self.comp_torque(s.q[0])
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


law = J1Law()
sess = TorqueSession(law, can=a.can)


def goto(q_target, secs=2.0):
    law.mode = "pin"
    law.u_t0 = None
    q0 = law.q_ref.copy() if law.q_ref is not None else sess.q()
    t0 = time.time()
    while time.time() - t0 < secs + 0.4:
        f = min((time.time() - t0) / secs, 1.0)
        law.q_ref = q0 + f * (q_target - q0)
        time.sleep(0.01)


print("stations (deg): %s  direction %s" % (np.degrees(stations).round(0), "+" if DIR > 0 else "-"))
print("ramp %.2f N.m/s, cap %.1f N.m, breakaway at %.1f deg motion" % (RATE, a.cap, BREAK_DEG))
if not a.yes:
    input(">>> CLEAR THE WORKSPACE - J1 sweeps %.0f..%.0f deg. ENTER to start " % (a.start, a.end))

rows = []
with sess:
    tgt = BASE.copy()
    tgt[0] = stations[0]
    goto(tgt, secs=3.0)
    for st in stations:
        tgt = law.q_ref.copy()
        tgt[0] = st
        goto(tgt, secs=max(0.8, abs(st - sess.q()[0]) / (15 * RAD)))
        time.sleep(0.6)                                   # settle pinned
        q_pin = sess.q()[0]
        law.mode = "ramp"                                 # free J1, start ramp
        u_break, jump = np.nan, np.nan
        while True:
            time.sleep(0.005)
            dq = (sess.q()[0] - q_pin) * DIR
            if dq > BREAK_DEG * RAD:
                u_break = law.u
                t0 = time.time()
                while time.time() - t0 < JUMP_S:          # let it slip, measure
                    jump = np.nanmax([jump, (sess.q()[0] - q_pin) * DIR])
                    time.sleep(0.005)
                break
            if law.u >= a.cap:
                break
            if not sess.running:
                raise SystemExit("session tripped: %s" % sess.trip)
        law.mode = "pin"                                  # catch it where it is
        law.u_t0 = None
        law.q_ref = sess.q()
        rows.append((q_pin, u_break, jump))
        print("  q1 %6.1f deg   breakaway %s   slip %s" % (
            np.degrees(q_pin),
            "%5.2f N.m" % u_break if np.isfinite(u_break) else ">cap ",
            "%4.2f deg" % np.degrees(jump) if np.isfinite(jump) else "  -  "))

    grid = np.array([r[0] for r in rows])
    taus = np.array([r[1] for r in rows])
    jumps = np.array([r[2] for r in rows])
    ok = np.isfinite(taus)
    if not ok.any():
        raise SystemExit("no breakaway below the %.1f N.m cap - raise --cap" % a.cap)
    print("\nJ1 stiction (%s direction): max %.2f  mean %.2f  min %.2f N.m   "
          "slip mean %.2f deg" % ("+" if DIR > 0 else "-", np.nanmax(taus),
                                  np.nanmean(taus[ok]), np.nanmin(taus[ok]),
                                  np.degrees(np.nanmean(jumps[ok]))))
    os.makedirs("data", exist_ok=True)
    np.savez("data/j1_stiction.npz", q1=grid, tau_break=taus, jump=jumps,
             direction=DIR, meta=dict(rate=RATE, cap=a.cap, base=BASE,
                                      date="2026-08-23"))
    print("saved data/j1_stiction.npz")

    # ---- compensation feel phase ----
    law.profile = (grid[ok], taus[ok])
    if not a.yes:
        input(">>> ENTER to feel J1 with alpha=%.2f compensation "
              "(push J1 %s; a/A adjusts, q quits) " % (law.alpha, "left" if DIR > 0 else "right"))
    goto_t = law.q_ref.copy()
    goto_t[0] = (a.start + 10.0 * DIR) * RAD              # restart near the sweep start
    goto(goto_t, secs=3.0)
    law.mode = "comp"
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        t_last = 0.0
        while sess.running:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1)
                if k == "q":
                    break
                elif k == "a":
                    law.alpha = max(0.0, law.alpha - 0.05)
                elif k == "A":
                    law.alpha = min(1.0, law.alpha + 0.05)
            if time.time() - t_last > 0.5:
                q1 = sess.q()[0]
                sys.stdout.write("\r\033[K alpha %.2f   q1 %6.1f deg   comp %5.2f N.m " % (
                    law.alpha, np.degrees(q1), law.comp_torque(q1)))
                sys.stdout.flush()
                t_last = time.time()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()
print("position hold restored" + (" (%s)" % sess.trip if sess.trip else ""))
