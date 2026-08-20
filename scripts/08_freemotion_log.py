"""Collect contact-free motion data for a NEXT-style torque estimator.

Follows the FACTR 2 protocol: sweep each joint independently across its range,
then multi-joint Cartesian-like motions, at both slow and fast speeds.

Logs, at the control rate, everything the estimator needs:
    q        measured joint positions
    qd       measured joint velocities
    q_cmd    the setpoint actually sent (so Delta_q_d = q_cmd - q is available)
    effort   SDK reported effort (motor current * fixed coefficient)

NOTHING MAY TOUCH THE ARM while this runs - every sample is labelled as
contact-free, and a single bump teaches the model that a push is normal.
"""
import argparse, sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm, JOINT_LIMITS
from piper_ht.model import PiperModel

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=10.0)
ap.add_argument("--rate", type=float, default=100.0, help="control/log rate Hz")
ap.add_argument("--speed", type=int, default=60)
ap.add_argument("--out", default="data/freemotion.npz")
a = ap.parse_args()

# Stay well inside the limits; these are the ranges the estimator will be valid over.
SAFE = np.radians(np.array([
    [-60.0, 60.0],    # J1
    [ 15.0, 85.0],    # J2
    [-135.0, -35.0],  # J3
    [-70.0, 70.0],    # J4
    [-45.0, 45.0],    # J5
    [-80.0, 80.0],    # J6
]))
CENTER = SAFE.mean(1)
AMP = (SAFE[:, 1] - SAFE[:, 0]) / 2
MIN_Z, MAX_R = 0.25, 0.58

mdl = PiperModel()
np.set_printoptions(precision=2, suppress=True, floatmode="fixed")


def safe(q):
    m = np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q)
    if m.min() < np.radians(5):
        return False
    p = mdl.fk(q)[:3, 3]
    return p[2] >= MIN_Z and np.hypot(p[0], p[1]) <= MAX_R


def clamp(q):
    return np.clip(q, SAFE[:, 0], SAFE[:, 1])


def build_segments(total_s):
    """(duration, trajectory-function) covering the FACTR 2 protocol."""
    segs = []
    # 1. each joint alone, slow then fast
    for j in range(6):
        for period in (8.0, 3.0):
            def f(t, j=j, period=period):
                q = CENTER.copy()
                q[j] = CENTER[j] + AMP[j] * 0.9 * np.sin(2 * np.pi * t / period)
                return q
            segs.append((period * 3, f))
    # 2. multi-joint Lissajous-style motion, slow then fast
    rng = np.random.default_rng(11)
    for scale, base in ((1.0, 11.0), (1.0, 6.0), (1.0, 3.5)):
        for _ in range(4):
            per = base * rng.uniform(0.75, 1.35, 6)
            pha = rng.uniform(0, 2 * np.pi, 6)
            amp = AMP * rng.uniform(0.45, 0.9, 6) * scale
            def f(t, per=per, pha=pha, amp=amp):
                return CENTER + amp * np.sin(2 * np.pi * t / per + pha)
            segs.append((base * 2.5, f))
    # 3. brief static holds, so the model also sees pure gravity
    for _ in range(6):
        qh = clamp(CENTER + AMP * rng.uniform(-0.8, 0.8, 6))
        segs.append((2.0, lambda t, qh=qh: qh))
    # repeat the whole programme until the requested duration is filled
    out, used = [], 0.0
    while used < total_s:
        for d, f in segs:
            if used >= total_s:
                break
            out.append((d, f)); used += d
    return out


arm = PiperArm().connect(0.5)
if not all(arm.is_enabled()):
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)

segs = build_segments(a.minutes * 60)
print("%d segments, ~%.1f min at %.0f Hz" % (len(segs), sum(d for d, _ in segs) / 60, a.rate))
print("!! The arm will move continuously through a large workspace.")
print("!! Keep the area clear and DO NOT TOUCH IT - every sample is labelled contact-free.")
input("\n>>> ENTER to begin ")

start = clamp(segs[0][1](0.0))
t0 = time.time()
while time.time() - t0 < 15:
    arm.move_j(start, speed_pct=20)
    if np.abs(arm.q() - start).max() < np.radians(1.5):
        break
    time.sleep(0.01)
time.sleep(0.5)

dt = 1.0 / a.rate
T, Q, QD, QC, EF = [], [], [], [], []
t_begin = time.time()
skipped = 0
try:
    for n, (dur, f) in enumerate(segs):
        s0 = time.time()
        while True:
            tt = time.time() - s0
            if tt >= dur:
                break
            q_des = clamp(f(tt))
            if not safe(q_des):
                skipped += 1
                q_des = QC[-1] if QC else start
            arm.move_j(q_des, speed_pct=a.speed)
            T.append(time.time() - t_begin)
            Q.append(arm.q()); QD.append(arm.dq())
            QC.append(np.asarray(q_des, float)); EF.append(arm.effort())
            time.sleep(max(0.0, dt - (time.time() - s0 - tt)))
        if n % 10 == 0:
            el = time.time() - t_begin
            print("  segment %3d/%d  elapsed %5.1f/%.0f s  samples %6d" %
                  (n + 1, len(segs), el, a.minutes * 60, len(T)))
except KeyboardInterrupt:
    print("\ninterrupted - saving what we have")

q_end = arm.q()
for _ in range(40):
    arm.move_j(q_end, speed_pct=10); time.sleep(0.01)

np.savez(a.out, t=np.array(T), q=np.array(Q), qd=np.array(QD),
         q_cmd=np.array(QC), effort=np.array(EF), rate=a.rate)
print("\nsaved %s: %d samples, %.1f min, %d unsafe poses skipped"
      % (a.out, len(T), (T[-1] if T else 0) / 60, skipped))
print("joint range covered (deg):")
Qa = np.array(Q)
for j in range(6):
    print("   J%d: %7.1f .. %7.1f" % (j + 1, np.degrees(Qa[:, j]).min(), np.degrees(Qa[:, j]).max()))
