"""Determine, per joint, the sign convention of reported effort.

Gravity cannot settle this for J1/J4/J6 (their gravity signature is negligible
or the URDF wrist model is wrong), so we use friction instead: friction always
opposes motion, so driving a joint in +q needs MORE motor torque than driving it
in -q.  If reported effort is higher while moving +, the reported sign agrees
with the joint-angle direction; if lower, it is inverted.

Small, slow, position-controlled motions.  Nobody should touch the arm.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm, JOINT_LIMITS
from piper_ht.model import PiperModel

BASE = np.radians([0.0, 45.0, -90.0, 0.0, 0.0, 0.0])
DELTA = np.radians([12.0, 8.0, 8.0, 15.0, 12.0, 20.0])
CYCLES = 3
SPEED = 12

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
mdl = PiperModel()

if not all(arm.is_enabled()):
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)

def goto(q, tol=1.0, timeout=14):
    t0 = time.time()
    while time.time() - t0 < timeout:
        arm.move_j(q, speed_pct=SPEED)
        if np.abs(arm.q() - q).max() < np.radians(tol):
            return True
        time.sleep(0.01)
    return False

print("moving to base pose", np.degrees(BASE))
goto(BASE, timeout=20)
time.sleep(0.8)

signs = np.zeros(6); conf = np.zeros(6)
for j in range(6):
    lo = BASE.copy(); hi = BASE.copy()
    lo[j] = BASE[j] - DELTA[j]; hi[j] = BASE[j] + DELTA[j]
    if (lo[j] < JOINT_LIMITS[j, 0] + np.radians(5) or
            hi[j] > JOINT_LIMITS[j, 1] - np.radians(5)):
        print("  J%d: range unsafe, skipped" % (j + 1)); continue
    up, dn = [], []
    goto(lo); time.sleep(0.4)
    for _ in range(CYCLES):
        # moving in the +q direction
        t0 = time.time(); arm.move_j(hi, speed_pct=SPEED)
        while np.abs(arm.q()[j] - hi[j]) > np.radians(1.5) and time.time() - t0 < 12:
            arm.move_j(hi, speed_pct=SPEED)
            if abs(arm.dq()[j]) > 0.02:
                up.append(arm.effort()[j])
            time.sleep(0.01)
        time.sleep(0.3)
        # moving in the -q direction
        t0 = time.time(); arm.move_j(lo, speed_pct=SPEED)
        while np.abs(arm.q()[j] - lo[j]) > np.radians(1.5) and time.time() - t0 < 12:
            arm.move_j(lo, speed_pct=SPEED)
            if abs(arm.dq()[j]) > 0.02:
                dn.append(arm.effort()[j])
            time.sleep(0.01)
        time.sleep(0.3)
    if len(up) < 20 or len(dn) < 20:
        print("  J%d: too few samples (%d/%d)" % (j + 1, len(up), len(dn))); continue
    mu_up, mu_dn = np.mean(up), np.mean(dn)
    gap = mu_up - mu_dn
    pooled = np.sqrt(np.var(up) + np.var(dn)) + 1e-9
    signs[j] = 1.0 if gap > 0 else -1.0
    conf[j] = abs(gap) / pooled
    print("  J%d: effort while +%.0fdeg = %+.3f | while -  = %+.3f | gap %+.3f | "
          "sign %+d (sep %.2f)" % (j + 1, np.degrees(DELTA[j]), mu_up, mu_dn, gap,
                                   int(signs[j]), conf[j]))
    goto(BASE)

goto(BASE)
np.savez("data/effort_sign.npz", sign=signs, confidence=conf)
print("\nSIGN =", signs)
print("confidence (gap / pooled sd):", conf.round(2))
print("\nsaved data/effort_sign.npz")
