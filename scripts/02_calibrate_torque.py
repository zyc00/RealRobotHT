"""Identify the joint torque model: scale, bias and Coulomb friction.

Each waypoint is visited twice, once sweeping up and once sweeping down.
  mean of the two directions -> gravity + bias   (friction cancels)
  half difference            -> Coulomb friction (gravity cancels)
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.model import PiperModel

SPEED = 15
SETTLE = 1.3
SAMPLE = 0.6
MIN_Z = 0.20

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
mdl = PiperModel()

sweep = sys.argv[1] if len(sys.argv) > 1 else "j2"
if sweep == "j2":
    poses = [np.radians([0, v, -90, 0, 0, 0]) for v in range(0, 81, 10)]
elif sweep == "j3":
    poses = [np.radians([0, 40, v, 0, 0, 0]) for v in range(-140, -19, 15)]
elif sweep == "j5":
    poses = [np.radians([0, 30, -90, 0, v, 0]) for v in range(-60, 61, 15)]
else:
    raise SystemExit("sweep must be j2|j3|j5")

for q in poses:
    z = mdl.fk(q)[2, 3]
    if z < MIN_Z:
        raise SystemExit("pose %s would put the tool at z=%.2f m, below the %.2f m floor" %
                         (np.degrees(q), z, MIN_Z))
print("sweep '%s': %d poses, tool height %.2f-%.2f m" % (
    sweep, len(poses), min(mdl.fk(q)[2,3] for q in poses), max(mdl.fk(q)[2,3] for q in poses)))


def goto(q, timeout=12.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        arm.move_j(q, speed_pct=SPEED)
        if np.abs(arm.q() - q).max() < np.radians(0.6):
            return True
        time.sleep(0.01)
    return False


def sample(q):
    E, Q, D = [], [], []
    t0 = time.time()
    while time.time() - t0 < SAMPLE:
        arm.move_j(q, speed_pct=SPEED)
        E.append(arm.effort()); Q.append(arm.q()); D.append(arm.dq())
        time.sleep(0.01)
    return np.array(Q).mean(0), np.array(E).mean(0), np.abs(np.array(D)).max()


print("\nmoving to start...")
goto(poses[0], timeout=20)
time.sleep(1.0)

rows = []
order = [("up", poses), ("down", poses[::-1])]
for direction, seq in order:
    print("\n--- sweeping %s ---" % direction)
    for q in seq:
        if not goto(q):
            print("  !! did not reach %s" % np.degrees(q)); continue
        time.sleep(SETTLE)
        q_m, tau_m, dq_max = sample(q)
        tau_g = mdl.gravity_torque(q_m)
        rows.append(dict(dir=direction, q=q_m, tau=tau_m, grav=tau_g))
        print("  q=%s  tau_meas=%s  tau_model=%s" % (
            np.degrees(q_m).round(1), tau_m.round(3), tau_g.round(3)))

np.savez("data/calib_%s.npz" % sweep,
         q=np.array([r["q"] for r in rows]),
         tau=np.array([r["tau"] for r in rows]),
         grav=np.array([r["grav"] for r in rows]),
         dirs=np.array([r["dir"] for r in rows]))
print("\nsaved data/calib_%s.npz  (%d samples)" % (sweep, len(rows)))
