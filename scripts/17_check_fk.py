"""Is our URDF FK actually wrong, or just measured at a different tool point?

If our kinematics are RIGHT and only the tool reference differs, then

    p_firmware(q) = p_fk(q) + R_fk(q) @ c        for one constant c

Solving that by least squares and looking at the residual tells us whether we
could run our own IK (and pick sane solution branches) instead of relying on
the firmware's.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm, JOINT_LIMITS
from piper_ht.model import PiperModel

N = int(sys.argv[1]) if len(sys.argv) > 1 else 14
mdl = PiperModel()
rng = np.random.default_rng(9)
np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


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

Q, P = [], []
for i, q in enumerate(poses):
    t0 = time.time()
    while time.time() - t0 < 14:
        arm.move_j(q, speed_pct=15)
        if np.abs(arm.q() - q).max() < np.radians(0.8):
            break
        time.sleep(0.01)
    time.sleep(0.8)
    ep = arm.piper.GetArmEndPoseMsgs().end_pose
    Q.append(arm.q())
    P.append(np.array([ep.X_axis, ep.Y_axis, ep.Z_axis]) * 1e-6)
    print("  %2d/%d  firmware %s" % (i + 1, N, P[-1].round(4)))

Q = np.array(Q); P = np.array(P)
np.savez("data/fk_check.npz", q=Q, p_fw=P)

# least squares for the constant tool offset c, expressed in the tool frame
A = np.zeros((3 * len(Q), 3)); b = np.zeros(3 * len(Q))
for i, q in enumerate(Q):
    T = mdl.fk(q)
    A[3 * i:3 * i + 3] = T[:3, :3]
    b[3 * i:3 * i + 3] = P[i] - T[:3, 3]
c, *_ = np.linalg.lstsq(A, b, rcond=None)
res = (A @ c - b).reshape(-1, 3)
err = np.linalg.norm(res, axis=1)

print("\nfitted tool offset c = %s m  (|c| = %.4f m)" % (c.round(4), np.linalg.norm(c)))
print("residual per pose: mean %.1f mm  max %.1f mm" % (err.mean() * 1000, err.max() * 1000))
if err.max() < 0.010:
    print("\n=> our FK is CORRECT; the firmware just reports a different tool point.")
    print("   We can run our own IK and choose sane solution branches.")
    np.savez("data/tool_offset.npz", c=c)
    print("   saved data/tool_offset.npz")
else:
    print("\n=> our FK does NOT match the arm (residual too large).")
    print("   Own-IK would inherit that error; stay with firmware IK.")
