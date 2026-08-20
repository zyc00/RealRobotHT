"""Drive the arm to the compact folded pose where a loss of torque is harmless."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm

np.set_printoptions(precision=2, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
target = np.radians([0, 5, -5, 0, 0, 0])
print("from:", np.degrees(arm.q()))
t0 = time.time()
while time.time() - t0 < 15:
    arm.move_j(target, speed_pct=15)
    if np.abs(arm.q() - target).max() < np.radians(1.0):
        break
    time.sleep(0.01)
time.sleep(0.5)
print("to  :", np.degrees(arm.q()))
print("effort:", arm.effort().round(3))
