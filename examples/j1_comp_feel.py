"""Feel J1 with the calibrated stiction compensated (single direction).

Loads data/j1_stiction.npz (from examples/j1_stiction.py), pins J2-J6 where
the arm stands, frees J1 and feeds forward

    tau1 = alpha * profile(q1) * dir        (zero within GUARD deg of range ends)

Push J1 in the calibrated direction (+ = left). Expected: breakaway needs
roughly (1-alpha) of the original force - the slip spike itself remains,
but the force you hold at break is much smaller, so less energy dumps in.
Pushing the OTHER way is uncompensated (heavier): single-direction test.

    python examples/j1_comp_feel.py                # alpha 0.8
    python examples/j1_comp_feel.py --alpha 0.9

  a / A  alpha -/+ 0.05      SPACE  compensation on/off (A/B!)      q  quit
  r      reset: PD-carries J1 back to the start position, then frees it again
  If J1 creeps by itself, alpha is above the local profile dip: press a.
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
PIN_KP, PIN_KD = 3.0, 0.8
GUARD_DEG = 6.0
BASE = np.radians([0.0, 45.0, -70.0, 0.0, 10.0, 0.0])   # calibration pose (J2-6)

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=0.8)
ap.add_argument("--data", default="data/j1_stiction.npz")
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

d = np.load(a.data, allow_pickle=True)
ok = np.isfinite(d["tau_break"])
GRID, TAUS = d["q1"][ok], d["tau_break"][ok]
DIR = float(d["direction"])
LO, HI = GRID.min() + GUARD_DEG * RAD, GRID.max() - GUARD_DEG * RAD
mdl = PiperModel()


class CompLaw:
    def __init__(self):
        self.q_ref = None
        self.alpha = a.alpha
        self.on = True
        self.comp = 0.0
        self.resetting = False
        self.t_prev = None
        self.dts = []

    def __call__(self, s):
        if self.t_prev is not None:
            self.dts.append(s.t - self.t_prev)
            if len(self.dts) > 3000:
                del self.dts[:1500]
        self.t_prev = s.t
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        tau = mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        if not self.resetting:
            kp[0] = kd[0] = 0.0                   # J1 free (pinned during reset)
        self.comp = 0.0
        if self.on and not self.resetting and LO < s.q[0] < HI:
            self.comp = self.alpha * np.interp(s.q[0], GRID, TAUS) * DIR
        tau[0] += self.comp
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


law = CompLaw()
sess = TorqueSession(law, can=a.can)
q1_home = float(sess.q()[0])
q0 = np.degrees(q1_home)
print("profile: %.2f..%.2f N.m over %.0f..%.0f deg, direction %s (push %s)"
      % (TAUS.min(), TAUS.max(), np.degrees(GRID.min()), np.degrees(GRID.max()),
         "+" if DIR > 0 else "-", "LEFT" if DIR > 0 else "RIGHT"))
print("J1 now at %.1f deg%s" % (q0, "" if LO < q0 * RAD < HI else "  (OUTSIDE range - no comp here!)"))
if not a.yes:
    input(">>> ENTER: J2-6 move to the CALIBRATION pose, then J1 goes free "
          "with alpha=%.2f comp (workspace clear!) " % a.alpha)


def goto_full(tgt, secs):
    """Slide the whole pin target while J1 is PD-held too."""
    law.resetting = True
    law.q_ref = sess.q()
    q0v, t0 = law.q_ref.copy(), time.time()
    while time.time() - t0 < secs + 0.4:
        f = min((time.time() - t0) / secs, 1.0)
        law.q_ref = q0v + f * (tgt - q0v)
        time.sleep(0.01)
    law.resetting = False

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)
try:
    with sess:
        tgt = BASE.copy()
        tgt[0] = q1_home
        goto_full(tgt, secs=3.0)          # J2-6 to the pose the profile was calibrated at
        t_last = 0.0
        while sess.running:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1)
                if k == "q":
                    break
                elif k == " ":
                    law.on = not law.on
                elif k == "r":
                    q1_now = float(sess.q()[0])
                    law.q_ref[0] = q1_now
                    law.resetting = True          # PD grabs J1 where it is
                    secs = max(0.8, abs(q1_home - q1_now) / (20 * RAD))
                    sys.stdout.write("\r\033[K resetting J1 to %+.1f deg (%.1f s)..." %
                                     (np.degrees(q1_home), secs))
                    sys.stdout.flush()
                    t0 = time.time()
                    while time.time() - t0 < secs + 0.5:
                        f = min((time.time() - t0) / secs, 1.0)
                        law.q_ref[0] = q1_now + f * (q1_home - q1_now)
                        time.sleep(0.01)
                    law.resetting = False         # free again, comp resumes
                elif k == "a":
                    law.alpha = max(0.0, law.alpha - 0.05)
                elif k == "A":
                    law.alpha = min(1.0, law.alpha + 0.05)
            if time.time() - t_last > 0.3:
                q1 = np.degrees(sess.q()[0])
                dts = np.array(law.dts[-1000:]) if law.dts else np.array([1.0])
                sys.stdout.write("\r\033[K %s  alpha %.2f  q1 %+6.1f deg  comp %+5.2f N.m  "
                                 "(local breakaway %.2f)  loop %3.0f Hz p99 %.1f ms" % (
                                     "COMP ON " if law.on else "comp off",
                                     law.alpha, q1, law.comp,
                                     np.interp(q1 * RAD, GRID, TAUS),
                                     1.0 / np.mean(dts), np.percentile(dts, 99) * 1e3))
                sys.stdout.flush()
                t_last = time.time()
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print()
print("position hold restored" + (" (%s)" % sess.trip if sess.trip else ""))
