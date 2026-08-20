"""Identify the firmware Euler convention WITHOUT trusting our URDF FK.

Rotating a single wrist joint produces a pure rotation about one fixed axis of
the tool frame.  So for the correct convention C:

    R_C(pose_i)^T @ R_C(pose_0)  is a rotation about a CONSTANT axis,
    through an angle equal to the joint's change.

Both facts are checkable from firmware data alone.  We sweep J6 (tool roll) and
J5, and score every candidate convention on how well it reproduces them.
"""
import sys, time, itertools
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm

BASE = np.radians([0.0, 45.0, -85.0, 0.0, 25.0, 0.0])
SWEEPS = [(5, np.arange(-70, 71, 20)), (4, np.arange(-35, 36, 15))]   # joint idx, degrees

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
if not all(arm.is_enabled()):
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)


def goto(q):
    t0 = time.time()
    while time.time() - t0 < 14:
        arm.move_j(q, speed_pct=15)
        if np.abs(arm.q() - q).max() < np.radians(0.8):
            return True
        time.sleep(0.01)
    return False


data = []
for jidx, degs in SWEEPS:
    print("sweeping J%d" % (jidx + 1))
    rows = []
    for d in degs:
        q = BASE.copy(); q[jidx] = np.radians(d)
        if not goto(q):
            continue
        time.sleep(0.8)
        ep = arm.piper.GetArmEndPoseMsgs().end_pose
        rpy = np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis]) * 1e-3
        rows.append((np.degrees(arm.q()[jidx]), rpy))
        print("   J%d=%+6.1f -> rpy %s" % (jidx + 1, rows[-1][0], rpy.round(1)))
    data.append((jidx, rows))
goto(BASE)

axis_of = {"x": 0, "y": 1, "z": 2}
seqs = ["".join(p) for p in itertools.permutations("xyz")]
seqs += ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"]
scored = []
for seq in seqs:
    for upper in (False, True):
        s = seq.upper() if upper else seq
        ang_err, axis_err, ok = [], [], True
        for jidx, rows in data:
            if len(rows) < 3:
                ok = False; break
            angs = np.stack([np.array([r[1][axis_of[c]] for c in seq]) for r in rows])
            try:
                R = Rot.from_euler(s, angs, degrees=True).as_matrix()
            except Exception:
                ok = False; break
            j0 = rows[0][0]
            axes = []
            for i in range(1, len(rows)):
                rel = Rot.from_matrix(R[0].T @ R[i])
                rv = rel.as_rotvec(degrees=True)
                mag = np.linalg.norm(rv)
                expect = abs(rows[i][0] - j0)
                ang_err.append(abs(mag - expect))
                if mag > 5:
                    axes.append(rv / mag)
            if len(axes) >= 2:
                A = np.array(axes)
                A *= np.sign(A @ A[0])[:, None]
                axis_err.append(np.degrees(np.arccos(np.clip(A @ A.mean(0) /
                                np.linalg.norm(A.mean(0)), -1, 1))).mean())
        if ok and ang_err:
            scored.append((np.mean(ang_err) + np.mean(axis_err or [0]),
                           np.mean(ang_err), np.mean(axis_err or [0]), s))

scored.sort()
print("\n  convention   angle-err   axis-spread   total")
for tot, ae, xe, s in scored[:8]:
    print("   %-6s     %7.2f     %7.2f     %7.2f" % (s, ae, xe, tot))
best = scored[0]
print("\n=> best '%s': angle error %.2f deg, axis spread %.2f deg" % (best[3], best[1], best[2]))
if best[1] < 5 and best[2] < 8:
    np.savez("data/euler_convention.npz", seq=best[3])
    print("   CONSISTENT - saved data/euler_convention.npz")
else:
    print("   still inconsistent - rotation stays locked")
