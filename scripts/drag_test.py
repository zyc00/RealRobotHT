"""Interactive drag test - you control the timing.

Moves to a test pose clear of all joint limits, captures a zero while you keep
your hands off, then prints a live external-torque estimate while you drag the
arm.  The arm stays in position hold the whole time, so it will resist you; we
are measuring how hard it fights back, not letting it move yet.

    ./scripts/drag_test.py           # default pose
    ./scripts/drag_test.py 40 -90    # J2, J3 in degrees
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm, JOINT_LIMITS
from piper_ht.model import PiperModel
from piper_ht.calibration import SCALE, BIAS

J2 = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
J3 = float(sys.argv[2]) if len(sys.argv) > 2 else -90.0
TEST_POSE = np.radians([0.0, J2, J3, 0.0, 0.0, 0.0])

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
mdl = PiperModel()

margin = np.degrees(np.minimum(TEST_POSE - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - TEST_POSE))
print("test pose (deg)      :", np.degrees(TEST_POSE).round(1))
print("margin to limits(deg):", margin.round(1))
if margin.min() < 8:
    sys.exit("!! too close to a joint limit - the end stop would absorb the load "
             "and corrupt the measurement. Pick another pose.")
print("tool height          : %.2f m" % mdl.fk(TEST_POSE)[2, 3])

if not all(arm.is_enabled()):
    print("\narm is not enabled; enabling and holding current pose first...")
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)

input("\n>>> Workspace clear? Press ENTER to move to the test pose ")
t0 = time.time()
while time.time() - t0 < 15:
    arm.move_j(TEST_POSE, speed_pct=15)
    if np.abs(arm.q() - TEST_POSE).max() < np.radians(0.8):
        break
    time.sleep(0.01)
time.sleep(0.8)
q_hold = arm.q()
print("arrived at:", np.degrees(q_hold).round(2))

def tau_true():
    return (arm.effort() - BIAS) / SCALE

input("\n>>> HANDS OFF the arm, then press ENTER to capture the zero ")
Z = []
t0 = time.time()
while time.time() - t0 < 2.0:
    arm.move_j(q_hold, speed_pct=10)
    Z.append(tau_true() - mdl.gravity_torque(arm.q()))
    time.sleep(0.01)
zero = np.array(Z).mean(0)
noise = np.array(Z).std(0)
print("zero  :", zero.round(3))
print("noise :", noise.round(4))

print("\n>>> NOW DRAG THE ARM. Push each joint both ways. Ctrl-C when done.\n")
print("      %8s %8s %8s %8s %8s %8s   %s" % ("J1", "J2", "J3", "J4", "J5", "J6", "peak|.|"))
log = []
peak = np.zeros(6)
try:
    n = 0
    while True:
        arm.move_j(q_hold, speed_pct=10)
        q = arm.q()
        ext = tau_true() - mdl.gravity_torque(q) - zero
        peak = np.maximum(peak, np.abs(ext))
        log.append((time.time(), q.copy(), ext.copy()))
        if n % 40 == 0:
            print("ext  %s   %s" % (" ".join("%8.3f" % v for v in ext), peak.round(2)))
        n += 1
        time.sleep(0.005)
except KeyboardInterrupt:
    pass

E = np.array([r[2] for r in log]); Q = np.array([r[1] for r in log])
np.savez("data/drag_test.npz", q=Q, ext=E, zero=zero, noise=noise, hold=q_hold)
print("\n\ncaptured %d samples" % len(E))
print("\n  joint     min      max   max|.|   noise    SNR")
for j in range(6):
    print("  j%d   %7.3f %7.3f %8.3f %7.4f %6.1f" % (
        j+1, E[:,j].min(), E[:,j].max(), np.abs(E[:,j]).max(), noise[j],
        np.abs(E[:,j]).max()/max(noise[j],1e-6)))
print("\nmax position deviation while dragged (deg):",
      np.degrees(np.abs(Q - q_hold).max(0)).round(3))
print("saved data/drag_test.npz")
