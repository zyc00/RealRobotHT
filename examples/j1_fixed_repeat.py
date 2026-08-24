"""Fixed-angle repeat test: is J1's stiction ripple degree- or time-dependent?

Pins J1 at ONE angle and measures breakaway many times. Position is held
constant, only time advances - so this removes the angle/time confound that a
monotonic sweep cannot.

Each trial logs three things:
  * u_break  - the breakaway torque
  * q_settle - the EXACT rest position before the ramp (feedback resolves
               0.001 deg; the soft pin lets the joint settle a hair off target,
               toward any local detent)
  * trial index (a proxy for time)

Interpretation:
  low  std(u_break)                         -> DEGREE-dependent (spatial): the
       spread we saw across a sweep was the ripple at different angles. Gear.
  high std but u_break tracks q_settle       -> spatial sub-degree ripple sampled
       (corr with position)                     at jittering rest positions.
  high std that trends with trial index      -> TIME-dependent (drift/thermal).
  high std, no correlation with either       -> temporal noise.

Runs at one angle by default; give several to see if the verdict holds across
the range (e.g. a known low at -40 vs a high at -34).

    python examples/j1_fixed_repeat.py                       # 0 deg, 25 reps
    python examples/j1_fixed_repeat.py --angle -40 --n 30
    python examples/j1_fixed_repeat.py --angles -40 -34 0    # a few angles
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
RATE = 0.12
PRELOAD = 0.30
BREAK_DEG = 0.7
JUMP_S = 0.2
REST_V = 0.015
REST_HOLD = 0.35

ap = argparse.ArgumentParser()
ap.add_argument("--angle", type=float, default=0.0, help="single test angle, deg")
ap.add_argument("--angles", type=float, nargs="+", default=None, help="several angles")
ap.add_argument("--n", type=int, default=25, help="repeats per angle")
ap.add_argument("--cap", type=float, default=1.5)
ap.add_argument("--can", default="can0")
ap.add_argument("--out", default="data/j1_fixed_repeat.npz")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()
ANGLES = a.angles if a.angles is not None else [a.angle]
mdl = PiperModel()


class J1Law:
    def __init__(self):
        self.mode = "pin"
        self.q_ref = None
        self.u = 0.0

    def __call__(self, s):
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        tau = mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        if self.mode == "ramp":
            kp[0] = kd[0] = 0.0
            tau[0] += self.u
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


law = J1Law()
sess = TorqueSession(law, can=a.can, hz=200.0)


def goto(q1, secs=1.2):
    law.mode = "pin"
    q0 = law.q_ref.copy() if law.q_ref is not None else sess.q()
    tgt = q0.copy(); tgt[0] = q1
    t0 = time.time()
    while time.time() - t0 < secs + 0.3:
        f = min((time.time() - t0) / secs, 1.0)
        law.q_ref = q0 + f * (tgt - q0)
        time.sleep(0.01)


def wait_at_rest(timeout=2.5):
    q_prev, t_prev, rest_since, t0 = sess.q()[0], time.time(), None, time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.02)
        q = sess.q()[0]
        v = abs(q - q_prev) / max(time.time() - t_prev, 1e-3)
        q_prev, t_prev = q, time.time()
        if v < REST_V:
            rest_since = rest_since or time.time()
            if time.time() - rest_since > REST_HOLD:
                return True
        else:
            rest_since = None
    return False


def one_trial(q_station):
    goto(q_station, secs=1.2)
    settled = wait_at_rest()
    q_settle = sess.q()[0]                       # exact rest position (0.001 deg)
    q_pin = q_settle
    law.u = PRELOAD
    law.mode = "ramp"
    t0 = time.time()
    ub, jump = np.nan, np.nan
    while True:
        time.sleep(0.004)
        law.u = min(PRELOAD + RATE * (time.time() - t0), a.cap)
        if (sess.q()[0] - q_pin) > BREAK_DEG * RAD:
            ub = law.u
            te = time.time()
            while time.time() - te < JUMP_S:
                jump = np.nanmax([jump, sess.q()[0] - q_pin]); time.sleep(0.004)
            break
        if law.u >= a.cap or not sess.running:
            break
    law.mode = "pin"; law.q_ref = sess.q()
    if not settled or (np.isfinite(ub) and ub < PRELOAD + 0.05 and
                       (not np.isfinite(jump) or jump < 2 * RAD)):
        ub = np.nan
    return ub, q_settle, np.degrees(jump) if np.isfinite(jump) else np.nan


print("fixed-angle repeat: angles %s deg, %d reps each" % (ANGLES, a.n))
if not a.yes:
    input(">>> CLEAR WORKSPACE, hand on E-STOP. ENTER to start ")

results = {}
with sess:
    goto(ANGLES[0], secs=3.0)
    for ang in ANGLES:
        print("\n=== angle %+.1f deg ===" % ang)
        us, qs, ts, order = [], [], [], []
        for i in range(a.n):
            ub, qset, slip = one_trial(ang * RAD)
            us.append(ub); qs.append(np.degrees(qset)); ts.append(time.time()); order.append(i)
            print("  #%2d  F_s %s N.m   q_settle %+7.3f deg   slip %s" % (
                i, "%.3f" % ub if np.isfinite(ub) else " nan ", np.degrees(qset),
                "%.1f" % slip if np.isfinite(slip) else "-"))
        us, qs, order = np.array(us), np.array(qs), np.array(order)
        results["a%+.0f" % ang] = dict(u=us, q_settle=qs, order=order, angle=ang)

goto_home = np.concatenate([[0.0], BASE[1:]])
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez(a.out, angles=ANGLES, n=a.n,
         **{"%s_%s" % (k, kk): v for k, d in results.items() for kk, v in d.items()})

print("\n" + "=" * 60)
for k, d in results.items():
    u, q, o = d["u"], d["q_settle"], d["order"]
    ok = np.isfinite(u)
    if ok.sum() < 4:
        print("angle %+.1f: too few valid trials (%d)" % (d["angle"], ok.sum())); continue
    u, q, o = u[ok], q[ok], o[ok]
    r_pos = np.corrcoef(u, q)[0, 1] if q.std() > 1e-4 else np.nan
    r_time = np.corrcoef(u, o)[0, 1]
    cv = u.std() / u.mean()
    if u.std() < 0.03:
        verdict = "DEGREE-dependent (reproducible at fixed angle -> gear/spatial)"
    elif abs(r_pos) > 0.5 and abs(r_pos) > abs(r_time):
        verdict = "SPATIAL sub-deg ripple (F_s tracks the %.3f deg settle jitter)" % q.std()
    elif abs(r_time) > 0.5:
        verdict = "TIME-dependent (F_s drifts with trial order -> thermal/drift)"
    else:
        verdict = "temporal NOISE (scatter, no position or time correlation)"
    print("angle %+.1f deg (n=%d):  F_s %.3f ± %.3f N.m (CV %.1f%%)" % (
        d["angle"], ok.sum(), u.mean(), u.std(), 100 * cv))
    print("   settle jitter %.3f deg | corr(F_s,pos) %+.2f | corr(F_s,time) %+.2f" % (
        q.std(), r_pos, r_time))
    print("   -> %s" % verdict)
print("saved %s" % a.out)
print("position hold restored" + (" (%s)" % sess.trip if sess.trip else ""))
