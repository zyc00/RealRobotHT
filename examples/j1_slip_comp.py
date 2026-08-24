"""Spike-killer feel test: stiction comp that STEPS DOWN to kinetic at slip.

Uses both calibrations (j1_stiction.npz + j1_kinetic.npz), J2-J6 pinned at
the calibration pose, J1 free, single direction. A state machine at --hz:

  STUCK : comp = alpha_s * F_s(q1)            (nearly all of local breakaway)
  SLIP  : comp = alpha_k * (f0(q1) + b*v)     (kinetic only)

Transition STUCK->SLIP fires when J1 moves BREAK_EPS off its rest anchor or
filtered velocity crosses V_SLIP; the feed-forward then drops by about
(F_s - f0) within one feedback sample (~5 ms), so the surplus that normally
becomes the spike is removed by control instead of accelerating the joint.
Re-stick after the joint rests V_RESTICK seconds.

Expected feel vs j1_comp_feel: same low breakaway force, but the jump at
break is largely gone - motion starts and STAYS at your fingertip pace.

    python examples/j1_slip_comp.py                 # 500 Hz loop
    python examples/j1_slip_comp.py --alpha-s 0.9 --alpha-k 0.8

  SPACE  comp on/off      a/A  alpha_s -/+.05    e/E  alpha_k -/+.05
                          (s/S and k/K still work as aliases)
  m      cycle mode: SLIPAWARE (state machine, discrete step-down) ->
         STATIC (constant alpha_s*F_s, = j1_comp_feel) ->
         STRIBECK (continuous: f0 + (F_s-f0)*exp(-|v|/V_STR) - no step at all,
         the assist melts from static to kinetic as speed rises)
  r      reset J1 to start    q    quit
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

# GIL convoy + GC are the measured latency-tail causes (p99 59->5.5 ms):
# see memory/control-loop-timing-measured.md
sys.setswitchinterval(0.0005)
import gc
gc.freeze()
gc.disable()

RAD = np.pi / 180.0
BASE = np.radians([0.0, 45.0, -70.0, 0.0, 10.0, 0.0])
PIN_KP, PIN_KD = 3.0, 0.8
GUARD_DEG = 6.0
BREAK_EPS = 0.01 * RAD      # slip trigger; feedback measured NOISELESS at 0.001deg
V_SLIP = 0.02               # rad/s, filtered
V_STICK = 0.008             # rad/s: below this counts as resting
V_RESTICK = 0.15            # s at rest before re-arming stiction comp
V_TAU = 0.010               # velocity filter, s
V_STR = 0.05                # rad/s, Stribeck blend width (mode 3)

ap = argparse.ArgumentParser()
ap.add_argument("--alpha-s", type=float, default=0.9)
ap.add_argument("--alpha-k", type=float, default=0.9)
ap.add_argument("--hz", type=float, default=500.0)
ap.add_argument("--stiction", default="data/j1_stiction.npz")
ap.add_argument("--kinetic", default="data/j1_kinetic.npz")
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

st = np.load(a.stiction, allow_pickle=True)
ok = np.isfinite(st["tau_break"])
GRID, TAUS = st["q1"][ok], st["tau_break"][ok]
DIR = float(st["direction"])
kin = np.load(a.kinetic, allow_pickle=True)
F0, B = float(kin["f0"]), float(kin["b"])
kok = np.isfinite(kin["f0_bins"])
KQ, KF = kin["q_bins"][kok], kin["f0_bins"][kok]
LO, HI = GRID.min() + GUARD_DEG * RAD, GRID.max() - GUARD_DEG * RAD
mdl = PiperModel()


def fs_of(q1):
    return np.interp(q1, GRID, TAUS)


def f0_of(q1):
    return np.interp(q1, KQ, KF) if len(KQ) > 2 else F0


class SlipLaw:
    def __init__(self):
        self.q_ref = None
        self.on = True
        self.resetting = False
        self.alpha_s, self.alpha_k = a.alpha_s, a.alpha_k
        self.mode = "SLIPAWARE"          # SLIPAWARE | STATIC | STRIBECK
        self.state = "STUCK"
        self.anchor = None
        self.v = 0.0
        self.rest_since = None
        self.comp = 0.0
        self.slips = 0
        self.t_prev, self.dts = None, []

    def __call__(self, s):
        if self.t_prev is not None:
            self.dts.append(s.t - self.t_prev)
            if len(self.dts) > 3000:
                del self.dts[:1500]
        dt = self.dts[-1] if self.dts else 1.0 / a.hz
        self.t_prev = s.t
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        q1 = s.q[0]
        self.v += (s.qdot[0] - self.v) * min(dt / V_TAU, 1.0)
        if self.anchor is None:
            self.anchor = q1

        # ---- state machine ----
        if self.state == "STUCK":
            if (q1 - self.anchor) * DIR > BREAK_EPS or self.v * DIR > V_SLIP:
                self.state = "SLIP"
                self.slips += 1
                self.rest_since = None
        else:
            if abs(self.v) < V_STICK:
                if self.rest_since is None:
                    self.rest_since = s.t
                elif s.t - self.rest_since > V_RESTICK:
                    self.state = "STUCK"
                    self.anchor = q1
            else:
                self.rest_since = None

        tau = mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        if not self.resetting:
            kp[0] = kd[0] = 0.0
        self.comp = 0.0
        if self.on and not self.resetting and LO < q1 < HI:
            if self.mode == "STRIBECK":
                blend = np.exp(-abs(self.v) / V_STR)
                self.comp = (self.alpha_k * (f0_of(q1) + B * max(self.v * DIR, 0.0))
                             + self.alpha_s * (fs_of(q1) - f0_of(q1)) * blend) * DIR
            elif self.state == "STUCK" or self.mode == "STATIC":
                self.comp = self.alpha_s * fs_of(q1) * DIR
            else:
                self.comp = self.alpha_k * (f0_of(q1) + B * max(self.v * DIR, 0.0)) * DIR
        tau[0] += self.comp
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


law = SlipLaw()
sess = TorqueSession(law, can=a.can, hz=a.hz)
q1_home = float(sess.q()[0])
print("F_s %.2f..%.2f N.m | kinetic f0 %.2f  b %.3f | drop at slip ~%.2f N.m | dir %s"
      % (TAUS.min(), TAUS.max(), F0, B,
         np.mean(fs_of(GRID)) * a.alpha_s - F0 * a.alpha_k, "+" if DIR > 0 else "-"))
if not a.yes:
    input(">>> ENTER: J2-6 to calibration pose, J1 free with slip-aware comp ")


def goto_full(tgt, secs):
    law.resetting = True
    law.q_ref = sess.q()
    q0v, t0 = law.q_ref.copy(), time.time()
    while time.time() - t0 < secs + 0.4:
        f = min((time.time() - t0) / secs, 1.0)
        law.q_ref = q0v + f * (tgt - q0v)
        time.sleep(0.01)
    law.anchor = None
    law.resetting = False


fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)
try:
    with sess:
        tgt = BASE.copy()
        tgt[0] = q1_home
        goto_full(tgt, secs=3.0)
        t_last = 0.0
        while sess.running:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1)
                if k == "q":
                    break
                elif k == " ":
                    law.on = not law.on
                elif k == "m":
                    order = ["SLIPAWARE", "STATIC", "STRIBECK"]
                    law.mode = order[(order.index(law.mode) + 1) % 3]
                elif k in ("a", "s"):
                    law.alpha_s = max(0.0, law.alpha_s - 0.05)
                elif k in ("A", "S"):
                    law.alpha_s = min(1.0, law.alpha_s + 0.05)
                elif k in ("e", "k"):
                    law.alpha_k = max(0.0, law.alpha_k - 0.05)
                elif k in ("E", "K"):
                    law.alpha_k = min(1.2, law.alpha_k + 0.05)
                elif k == "r":
                    tgt = law.q_ref.copy()
                    tgt[0] = q1_home
                    goto_full(tgt, secs=max(0.8, abs(q1_home - sess.q()[0]) / (20 * RAD)))
            if time.time() - t_last > 0.3:
                dts = np.array(law.dts[-2000:]) if law.dts else np.array([1.0])
                q1 = np.degrees(sess.q()[0])
                sys.stdout.write("\r\033[K %s %s %-5s a_s %.2f a_k %.2f  q1 %+6.1f  "
                                 "comp %+5.2f N.m  slips %d  loop %3.0f Hz p99 %.1f ms" % (
                                     "ON " if law.on else "off",
                                     "%-9s" % law.mode,
                                     law.state,
                                     law.alpha_s, law.alpha_k, q1, law.comp, law.slips,
                                     1.0 / np.mean(dts), np.percentile(dts, 99) * 1e3))
                sys.stdout.flush()
                t_last = time.time()
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print()
print("position hold restored" + (" (%s)" % sess.trip if sess.trip else ""))
