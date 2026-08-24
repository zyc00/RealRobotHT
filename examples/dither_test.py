"""Stick-slip feel test: dither, friction EQUALIZER and damping, A/B by key.

Gravity-compensated drag mode (as drag_mode.py) plus three switchable terms:

    tau_j = G_j(q) + A_j sin(2 pi f t + phi_j)                      dither
                   - E_j tanh(v_j/V_EPS) exp(-|v_j|/V_FADE)          equalizer
                   - B_j v_j                                         damping

The 1 mm -> 2 mm jump when hand-guiding is stick-slip: force ramps to static
friction F_s, then friction drops to F_k and the surplus (F_s - F_k) stored in
your arm accelerates the joint. The EQUALIZER adds Coulomb braking of about
(F_s - F_k) that fades out above V_FADE, so the joint starts moving at the
same force it broke away at: uniformly heavy, but no surplus, no jump.
Damping limits the jump velocity instead (heavier while moving).

Dither above the arm's mechanical bandwidth keeps the joints in the kinetic
friction regime, so breakaway effort should drop. Toggle it on/off while
dragging and judge by feel; the status line reports achieved loop rate,
samples per dither period and joint-velocity RMS (does it visibly buzz?).

Default amplitudes are the measured KINETIC friction per joint
(data/friction_fit.npz via balanced_drag.py): J1 .32 J2 .60 J3 .29 J4 .09
J5 .07 J6 .05 N.m. Static friction is higher, so you may need 1.5-2x.

    python examples/dither_test.py                   # 500 Hz loop, 40 Hz dither
    python examples/dither_test.py --hz 1000 --freq 60
    python examples/dither_test.py --joints 1 2 3    # base joints only
    python examples/dither_test.py --joints 1 2 3 --cap 2 --freq 15   # reach gear-side

  SPACE  dither on/off (A/B)     f / F  freq down / up (1 Hz steps below 10)
  a / A  amplitude x0.8 / x1.25  1..6   toggle a joint      q  quit
  e      equalizer on/off        E      equalizer scale cycle .5 1 1.5 2 3
  d      damping on/off          D      damping scale cycle .5 1 1.5 2 3
  s      SWEEP: 1 -> 80 Hz over ~40 s at current amplitude, logs per-joint
         velocity response per band -> the CAN->motor torque bandwidth
  p      PULSE: 0.3 s step of the current amplitude on the active joints
         (sanity check that torque reaches the motors at all)

Sanity path if you feel nothing: press f until 2 Hz and A twice - at 2 Hz the
arm MUST visibly wobble if the torque path works. Then F upward and watch
qdot rms fall: where it vanishes is the bandwidth. Below ~20 Hz dither is
useless (it IS the motion); no response anywhere = firmware filters t_ff.

The arm buzzes; hold it. Watchdogs from TorqueSession stay active.
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

F0 = np.array([0.32, 0.60, 0.29, 0.09, 0.07, 0.05])       # kinetic friction, N.m
AMP_MAX = np.array([1.0, 1.5, 1.0, 0.5, 0.4, 0.3])         # hard cap per joint, N.m
DROP = np.array([0.15, 0.50, 0.20, 0.12, 0.08, 0.05])      # ~F_s - F_k guess, N.m (scale with E)
EQ_MAX = np.array([0.8, 1.5, 0.8, 0.4, 0.3, 0.2])
B0 = np.array([0.5, 0.8, 0.5, 0.15, 0.10, 0.08])           # damping, N.m.s/rad (scale with D)
V_EPS, V_FADE, V_TAU = 0.02, 0.3, 0.02                     # rad/s, rad/s, velocity filter s
SCALES = [0.5, 1.0, 1.5, 2.0, 3.0]

ap = argparse.ArgumentParser()
ap.add_argument("--hz", type=float, default=500.0, help="control loop rate")
ap.add_argument("--freq", type=float, default=40.0, help="dither frequency, Hz")
ap.add_argument("--scale", type=float, default=1.0, help="amplitude = scale * F0")
ap.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
ap.add_argument("--cap", type=float, default=1.0,
                help="multiply AMP_MAX (1.0/1.5/1.0/.5/.4/.3 N.m); reaching gear-side "
                     "breakaway on J2 needs ~2-3 N.m -> --cap 2. Low freq + big amp SHAKES.")
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()
AMP_MAX = AMP_MAX * a.cap

mdl = PiperModel()


class DitherLaw:
    def __init__(self):
        self.on = False
        self.freq = a.freq
        self.scale = a.scale
        self.mask = np.zeros(6)
        for j in a.joints:
            self.mask[j - 1] = 1.0
        self.phase = np.arange(6) * 2 * np.pi / 6          # stagger joints
        # stats (written by the session thread, read by main)
        self.t_prev = None
        self.dts = []
        self.v2 = np.zeros(6)
        self.n = 0
        self.pulse_until = -1.0
        self.sweep = None                 # (t0, f_lo, f_hi, secs) while sweeping
        self.eq_on, self.eq_scale = False, 1.0
        self.damp_on, self.damp_scale = False, 1.0
        self.v = np.zeros(6)              # filtered velocity (raw qdot is 200 Hz fd at 500 Hz)

    def amp(self):
        return np.minimum(self.scale * F0, AMP_MAX) * self.mask

    def __call__(self, s):
        if self.t_prev is not None:
            self.dts.append(s.t - self.t_prev)
            if len(self.dts) > 4000:
                del self.dts[:2000]
        self.t_prev = s.t
        self.v2 += s.qdot ** 2
        self.n += 1
        tau = mdl.gravity_torque(s.q)
        if self.sweep is not None:
            t0, f_lo, f_hi, secs = self.sweep
            u = min((s.t - t0) / secs, 1.0)
            self.freq = f_lo * (f_hi / f_lo) ** u           # log sweep
            if u >= 1.0:
                self.sweep = None
        dt = self.dts[-1] if self.dts else 1.0 / a.hz
        self.v += (s.qdot - self.v) * min(dt / V_TAU, 1.0)
        if self.on:
            tau = tau + self.amp() * np.sin(2 * np.pi * self.freq * s.t + self.phase)
        if self.eq_on:
            E = np.minimum(self.eq_scale * DROP, EQ_MAX)
            tau = tau - E * np.tanh(self.v / V_EPS) * np.exp(-np.abs(self.v) / V_FADE)
        if self.damp_on:
            tau = tau - self.damp_scale * B0 * self.v
        if s.t < self.pulse_until:
            tau = tau + self.amp()
        return MitCommand(t_ff=tau)

    def joint_rms(self):
        v = np.sqrt(self.v2 / max(self.n, 1))
        self.v2, self.n = np.zeros(6), 0
        return v

    def status(self):
        dts = np.array(self.dts[-2000:]) if self.dts else np.array([1.0 / a.hz])
        rate = 1.0 / np.mean(dts)
        v = self.joint_rms()
        return ("%s f %5.1f Hz amp x%.2f | EQ %s x%.1f | DAMP %s x%.1f | loop %4.0f Hz "
                "(p99 %.1f ms) %.0f samp/per | qdot rms [%s]" % (
                    "DITHER ON " if self.on else "dither off",
                    self.freq, self.scale,
                    "ON " if self.eq_on else "off", self.eq_scale,
                    "ON " if self.damp_on else "off", self.damp_scale,
                    rate, np.percentile(dts, 99) * 1e3, rate / self.freq,
                    " ".join("%.2f" % x for x in v)))


law = DitherLaw()
sess = TorqueSession(law, can=a.can, hz=a.hz)
print("pose (deg):", np.degrees(sess.q()).round(1))
print("dither amplitudes (N.m):", law.amp().round(2), "at %.0f Hz, loop %.0f Hz" % (a.freq, a.hz))
if not a.yes:
    input(">>> ENTER to go compliant (all terms start OFF: SPACE dither, e equalizer, d damping) ")

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)
try:
    with sess:
        t_last = 0.0
        while sess.running:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1)
                if k == "q":
                    break
                elif k == " ":
                    law.on = not law.on
                elif k == "f":
                    law.freq = max(1.0, law.freq - (1.0 if law.freq <= 10 else 5.0))
                elif k == "F":
                    law.freq = min(a.hz / 4.0, law.freq + (1.0 if law.freq < 10 else 5.0))
                elif k == "p":
                    law.pulse_until = (law.t_prev or 0.0) + 0.3
                elif k == "s":
                    law.on = True
                    law.joint_rms()
                    law.sweep = ((law.t_prev or 0.0), 1.0, 80.0, 40.0)
                    sys.stdout.write("\r\033[K sweep 1->80 Hz, amp x%.2f\r\n" % law.scale)
                    f_last = None
                    while law.sweep is not None and sess.running:
                        time.sleep(0.05)
                        band = int(np.log2(max(law.freq, 1.0)) * 2) / 2   # half-octaves
                        if band != f_last:
                            if f_last is not None:
                                sys.stdout.write("   %5.1f Hz  qdot rms [%s]\r\n" % (
                                    2 ** f_last, " ".join("%.3f" % x for x in law.joint_rms())))
                                sys.stdout.flush()
                            else:
                                law.joint_rms()
                            f_last = band
                    sys.stdout.write("   sweep done (dither left ON at %.0f Hz)\r\n" % law.freq)
                elif k == "e":
                    law.eq_on = not law.eq_on
                elif k == "E":
                    law.eq_scale = SCALES[(SCALES.index(law.eq_scale) + 1) % len(SCALES)]
                elif k == "d":
                    law.damp_on = not law.damp_on
                elif k == "D":
                    law.damp_scale = SCALES[(SCALES.index(law.damp_scale) + 1) % len(SCALES)]
                elif k == "a":
                    law.scale = max(0.1, law.scale * 0.8)
                elif k == "A":
                    law.scale = min(4.0, law.scale * 1.25)
                elif k in "123456":
                    j = int(k) - 1
                    law.mask[j] = 0.0 if law.mask[j] else 1.0
                else:
                    continue
                sys.stdout.write("\r\033[K" + law.status() + "\r\n")
                t_last = time.time()
            if time.time() - t_last > 1.0:
                sys.stdout.write("\r\033[K" + law.status())
                sys.stdout.flush()
                t_last = time.time()
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print()
print("position hold restored" + (" (%s)" % sess.trip if sess.trip else ""))
