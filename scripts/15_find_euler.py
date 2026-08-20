"""Identify the firmware's end-pose Euler convention.

The SDK calls RX/RY/RZ "Euler angles" without saying which of the 24 possible
conventions.  We can find out: for the correct convention C,

    R_C(RX,RY,RZ)  =  R_fk(q) @ R_offset

with a CONSTANT R_offset (our URDF tool frame may differ from the firmware's by
a fixed rotation).  So build R from the reported angles under each candidate
convention, and pick the one where R_fk^T @ R_C is the same at every pose.
"""
import sys, time, itertools
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm, JOINT_LIMITS
from piper_ht.model import PiperModel

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
mdl = PiperModel()
rng = np.random.default_rng(5)


def safe(q):
    m = np.degrees(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q))
    if m.min() < 8:
        return False
    p = mdl.fk(q)[:3, 3]
    return p[2] > 0.28 and np.hypot(p[0], p[1]) < 0.5


poses = []
while len(poses) < N:
    q = np.radians([rng.uniform(-40, 40), rng.uniform(25, 70), rng.uniform(-110, -50),
                    rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(-70, 70)])
    if safe(q):
        poses.append(q)

arm = PiperArm().connect(0.5)
if not all(arm.is_enabled()):
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)

print("visiting %d poses (wrist-heavy, to separate the conventions)" % N)
Q, E = [], []
for i, q in enumerate(poses):
    t0 = time.time()
    while time.time() - t0 < 14:
        arm.move_j(q, speed_pct=15)
        if np.abs(arm.q() - q).max() < np.radians(0.8):
            break
        time.sleep(0.01)
    time.sleep(0.9)
    ep = arm.piper.GetArmEndPoseMsgs().end_pose
    Q.append(arm.q())
    E.append(np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis]) * 1e-3)
    print("  %2d/%d q=%s  rpy=%s" % (i + 1, N, np.degrees(Q[-1]).round(0), E[-1].round(1)))

Q = np.array(Q); E = np.array(E)
np.savez("data/euler_ident.npz", q=Q, rpy=E)

R_fk = [mdl.fk(q)[:3, :3] for q in Q]
axis_of = {"x": 0, "y": 1, "z": 2}
results = []
seqs = ["".join(p) for p in itertools.permutations("xyz")]
seqs += ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"]
for seq in seqs:
    for upper in (False, True):          # lower = extrinsic, upper = intrinsic
        s = seq.upper() if upper else seq
        try:
            ang = np.stack([E[:, axis_of[c]] for c in seq], axis=1)
            R_c = Rot.from_euler(s, ang, degrees=True).as_matrix()
        except Exception:
            continue
        offs = [R_fk[i].T @ R_c[i] for i in range(len(Q))]
        mean = np.mean(offs, axis=0)
        u, _, vt = np.linalg.svd(mean)
        M = u @ vt
        spread = np.mean([np.degrees(np.linalg.norm(Rot.from_matrix(M.T @ o).as_rotvec()))
                          for o in offs])
        results.append((spread, s, M))

results.sort(key=lambda r: r[0])
print("\nbest-matching conventions (lower spread = more consistent):")
for spread, s, _ in results[:6]:
    print("   %-5s  offset spread %7.2f deg" % (s, spread))
best = results[0]
print("\n=> convention '%s', residual %.2f deg" % (best[1], best[0]))
if best[0] < 3.0:
    print("   CONSISTENT - usable")
    np.savez("data/euler_convention.npz", seq=best[1], offset=best[2])
    print("   saved data/euler_convention.npz  (offset rotation included)")
else:
    print("   NOT consistent - no standard Euler convention explains the data")
