"""Dense J1 static-friction sweep - resolve the ~20 deg ripple.

The 5 deg map showed a periodic dip in breakaway torque (~20 deg period, 0.1
N.m). At 5 deg that is only ~4 samples/period - could be a finer transmission
ripple (gear mesh / cogging) aliased up. This samples finely (default 1.5 deg)
to reveal the true period and shape. Also runs an FFT and prints the dominant
spatial period(s).

Each station VERIFIES J1 is at rest before ramping (a breakaway slips ~9 deg;
at fine spacing the joint must re-settle first or it false-triggers at the
preload torque - the 1st dense run's fake 2 deg ripple was exactly that).
False triggers (tiny slip at ~preload) are rejected to NaN. ~6-7 s/station.

J1 is a vertical axis (gravity ~0), so a single direction is clean. --both
averages +/- to cancel any residual and reports directional asymmetry.

THE ARM SWEEPS J1 REPEATEDLY BY ITSELF: CLEAR THE WORKSPACE.

    python examples/j1_stiction_dense.py --step 2.0      # 2 deg, ~6 min (start here)
    python examples/j1_stiction_dense.py                 # 1.5 deg, ~8 min
    python examples/j1_stiction_dense.py --both          # both directions
"""
import argparse
import os
import sys
import time

_PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
try:
    from piperx_teleop import MitCommand, PiperModel, TorqueSession, require_patched_sdk
    require_patched_sdk()
except (RuntimeError, ImportError):
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

import numpy as np

sys.setswitchinterval(0.0005)
RAD = np.pi / 180.0
BASE = np.radians([0.0, 45.0, -70.0, 0.0, 10.0, 0.0])
PIN_KP, PIN_KD = 3.0, 0.8
RATE = 0.12               # N.m/s ramp
PRELOAD = 0.30            # N.m ramp start (well below min breakaway)
BREAK_DEG = 0.7           # real breakaway slips ~9 deg; small thresh false-triggers
JUMP_S = 0.25
REST_V = 0.015            # rad/s: below this = at rest
REST_HOLD = 0.3           # s the joint must stay at rest before ramping

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=float, default=-52.0)
ap.add_argument("--end", type=float, default=48.0)
ap.add_argument("--step", type=float, default=1.5, help="deg between stations")
ap.add_argument("--cap", type=float, default=1.5)
ap.add_argument("--both", action="store_true", help="sweep + and - and average")
ap.add_argument("--can", default="can0")
ap.add_argument("--out", default="data/j1_stiction_dense.npz")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

grid = np.arange(a.start, a.end + 1e-6, a.step) * RAD
mdl = PiperModel()


class J1Law:
    def __init__(self):
        self.mode = "pin"           # pin | ramp
        self.q_ref = None
        self.u = 0.0
        self.direction = 1.0

    def __call__(self, s):
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        tau = mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        if self.mode == "ramp":
            kp[0] = kd[0] = 0.0
            tau[0] += self.u * self.direction
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


law = J1Law()
sess = TorqueSession(law, can=a.can, hz=200.0)


def goto(q1_target, secs=1.0):
    law.mode = "pin"
    q0 = law.q_ref.copy() if law.q_ref is not None else sess.q()
    tgt = q0.copy(); tgt[0] = q1_target
    t0 = time.time()
    while time.time() - t0 < secs + 0.3:
        f = min((time.time() - t0) / secs, 1.0)
        law.q_ref = q0 + f * (tgt - q0)
        time.sleep(0.01)


def wait_at_rest(timeout=2.5):
    """Block until J1 has held still for REST_HOLD s (settled from prior slip)."""
    q_prev, t_prev = sess.q()[0], time.time()
    rest_since = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.02)
        q = sess.q()[0]
        v = abs(q - q_prev) / max(time.time() - t_prev, 1e-3)
        q_prev, t_prev = q, time.time()
        if v < REST_V:
            if rest_since is None:
                rest_since = time.time()
            elif time.time() - rest_since > REST_HOLD:
                return True
        else:
            rest_since = None
    return False


def breakaway(q1_station, direction):
    """Verify at rest, ramp from PRELOAD until J1 moves BREAK_DEG; (u_break, slip)."""
    settled = wait_at_rest()
    q_pin = sess.q()[0]
    law.direction = direction
    law.u = PRELOAD
    law.mode = "ramp"
    t0 = time.time()
    u_break, jump = np.nan, np.nan
    while True:
        time.sleep(0.004)
        law.u = min(PRELOAD + RATE * (time.time() - t0), a.cap)
        if (sess.q()[0] - q_pin) * direction > BREAK_DEG * RAD:
            u_break = law.u
            te = time.time()
            while time.time() - te < JUMP_S:
                jump = np.nanmax([jump, (sess.q()[0] - q_pin) * direction])
                time.sleep(0.004)
            break
        if law.u >= a.cap or not sess.running:
            break
    law.mode = "pin"
    law.q_ref = sess.q()
    slip = np.degrees(jump) if np.isfinite(jump) else np.nan
    # reject false triggers: real breakaway slips several deg; tiny slip at
    # ~preload means the joint wasn't settled -> mark invalid
    if np.isfinite(u_break) and u_break < PRELOAD + 0.05 and (not np.isfinite(slip) or slip < 2.0):
        u_break = np.nan
    if not settled:
        u_break = np.nan
    return u_break, slip


def sweep(direction):
    qs, us, slips = [], [], []
    ordered = grid if direction > 0 else grid[::-1]
    for k, st in enumerate(ordered):
        goto(st, secs=max(0.6, abs(st - sess.q()[0]) / (14 * RAD)))
        ub, sl = breakaway(st, direction)
        qs.append(st); us.append(ub); slips.append(sl)
        if k % 10 == 0 or not np.isfinite(ub):
            print("   %+6.1f deg  F_s %s N.m" % (np.degrees(st),
                  "%.2f" % ub if np.isfinite(ub) else ">cap"))
    order = np.argsort(qs)
    return np.array(qs)[order], np.array(us)[order], np.array(slips)[order]


print("dense J1 stiction: %d stations %.1f..%.1f deg, step %.1f deg%s" % (
    len(grid), a.start, a.end, a.step, ", BOTH dir" if a.both else ""))
print("preload %.2f N.m, ramp %.2f N.m/s" % (PRELOAD, RATE))
if not a.yes:
    input(">>> CLEAR WORKSPACE, hand on E-STOP. ENTER to sweep ")

with sess:
    goto(grid[0], secs=3.0)
    qp, up, slp = sweep(+1.0)
    if a.both:
        qm, um, slm = sweep(-1.0)
        fs = np.nanmean([up, np.interp(qp, qm, um)], axis=0)
        asym = up - np.interp(qp, qm, um)
    else:
        qm = um = slm = None
        fs = up; asym = np.full_like(up, np.nan)
    goto(0.0, secs=3.0)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez(a.out, q=qp, F_s=fs, F_s_plus=up, slip_plus=slp, F_s_minus=um,
         q_minus=qm, asym=asym, step=a.step, date="2026-08-23")

# --- spectral analysis of the ripple ---
ok = np.isfinite(fs)
q_ok, f_ok = np.degrees(qp[ok]), fs[ok]
detr = f_ok - np.polyval(np.polyfit(q_ok, f_ok, 1), q_ok)
qu = np.linspace(q_ok.min(), q_ok.max(), len(q_ok))       # uniform for FFT
fu = np.interp(qu, q_ok, detr)
sp = np.abs(np.fft.rfft(fu - fu.mean()))
fr = np.fft.rfftfreq(len(fu), qu[1] - qu[0])              # cycles/deg
peaks = np.argsort(sp[1:])[::-1][:3] + 1
print("\n== dense J1 stiction ==")
print("mean F_s %.3f  std %.3f  ptp %.3f N.m  (%d stations)" % (
    f_ok.mean(), f_ok.std(), np.ptp(f_ok), len(f_ok)))
print("linear R^2 %.3f (flat if ~0)" % (
    1 - np.var(detr) / max(np.var(f_ok), 1e-9)))
print("dominant spatial periods:")
for k in peaks:
    if fr[k] > 0:
        print("   %.1f deg/cycle   amplitude %.3f N.m" % (1 / fr[k], 2 * sp[k] / len(fu)))
print("saved %s" % a.out)
print("position hold restored" + (" (%s)" % sess.trip if sess.trip else ""))
