"""Compliant control: a virtual spring-damper on top of gravity compensation.

This is what the old admittance controller was always trying to be. Admittance
(infer the push from motor current, walk a position setpoint) was the
workaround for having no torque interface - deadbands, hysteresis, stutter.
With torque control the same goal is IMPEDANCE, and it needs none of that:

    tau = G(q) + Kp*(q_ref - q) - Kd*qdot      # PD runs in the firmware

Push the arm: it yields and springs back. Kp=0 is drag mode. In between is a
compliant hold - and a compliant FOLLOWER, if you stream a leader's pose into
set_target().

    python examples/impedance_demo.py                 # hold here, springy
    python examples/impedance_demo.py --kp 20         # stiffer spring
    python examples/impedance_demo.py --wander        # compliant waypoint tour

From code:

    from piperx_teleop import JointImpedance
    with JointImpedance(kp=8.0) as imp:     # holds current pose compliantly
        imp.move_to(q_goal, secs=2.0)       # compliant motion
        imp.set_target(leader.q())          # or track a leader, softly
"""
import argparse
import os
import sys
import time

_PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
try:
    from piperx_teleop import JointImpedance, require_patched_sdk
    require_patched_sdk()
except (RuntimeError, ImportError):
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

import numpy as np

ap = argparse.ArgumentParser(description="joint impedance: springy compliant hold")
ap.add_argument("--kp", type=float, default=None,
                help="stiffness, all joints (default: per-joint [15,15,8,5,5,2])")
ap.add_argument("--kd", type=float, default=0.8, help="damping")
ap.add_argument("--wander", action="store_true",
                help="tour a few waypoints compliantly instead of holding")
ap.add_argument("--can", default="can0")
a = ap.parse_args()

kp = a.kp if a.kp is not None else [15.0, 15.0, 8.0, 5.0, 5.0, 2.0]
imp = JointImpedance(can=a.can, kp=kp, kd=a.kd)
q0 = imp.q()
print("pose (deg):", np.degrees(q0).round(1))
print("kp:", imp.kp.round(1), " kd:", imp.kd.round(2))
input(">>> ENTER to go compliant - push the arm and it springs back ")

with imp:
    try:
        if a.wander:
            base = np.radians([0, 45, -75, 0, 15, 0])
            imp.move_to(base, secs=2.5)
            offsets = [[15, -10, 10, 20, -15, 30], [-15, 10, -15, -20, 15, -30],
                       [0, 15, 5, 0, 20, 0], [0, 0, 0, 0, 0, 0]]
            while imp.running:
                for off in offsets:
                    if not imp.running:
                        break
                    imp.move_to(base + np.radians(off), secs=2.0)
                    time.sleep(0.5)
        else:
            while imp.running:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
print("position hold restored" + (" (%s)" % imp.trip if imp.trip else ""))
