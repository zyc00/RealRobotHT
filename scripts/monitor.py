"""Passive state monitor. Does not command the arm - safe to run any time."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.model import PiperModel

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
arm = PiperArm().connect()
mdl = PiperModel()
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

print("enabled:", arm.is_enabled())
print(f"{'t':>5} {'q (deg)':>46} {'dq (rad/s)':>40} {'|tau| meas (N.m)':>40} {'tau grav model (N.m)':>40}")
t0 = time.time()
while time.time() - t0 < dur:
    s = arm.state()
    g = mdl.gravity_torque(s["q"])
    print(f"{time.time()-t0:5.1f} {np.degrees(s['q'])} {s['dq']} {s['effort']} {g}")
    time.sleep(0.25)
