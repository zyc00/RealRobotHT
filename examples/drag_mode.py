"""Drag mode: gravity-compensated compliance from the piperx_teleop package.

The arm becomes freely draggable and holds wherever you leave it. Pure
torque-mode gravity feed-forward with the PiPER-X model - no gains to tune.

    python examples/drag_mode.py                  # drag until Ctrl-C
    python examples/drag_mode.py --duration 20
    piperx-drag                                   # same thing, installed CLI

From your own code (recording, leader-follower teleop):

    from piperx_teleop import GravityCompensator

    with GravityCompensator() as gc:              # arm goes compliant
        while collecting:
            log(gc.q(), gc.gripper())             # 50 Hz reads, your loop
    # position hold restored here; gc.trip explains any abnormal stop

    leader = GravityCompensator(can="can0")       # leader-follower skeleton
    follower = PiperArm("can1").connect()
    with leader:
        while True:
            follower.move_j(leader.q())
"""
import argparse
import os
import sys

# Needs the patched piper_sdk (12-bit MIT frame); re-exec into piperctl if
# this env has the stock one.
_PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
try:
    from piperx_teleop import GravityCompensator, require_patched_sdk
    require_patched_sdk()
except (RuntimeError, ImportError):
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

import numpy as np

ap = argparse.ArgumentParser(description="gravity-compensated drag mode")
ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
ap.add_argument("--can", default="can0")
ap.add_argument("--payload-mass", type=float, default=0.0, help="kg on the gripper")
ap.add_argument("--tool", default=None, help="tool file from examples/tool_id.py (mass+COM past joint 6)")
ap.add_argument("--friction", default=None, help="friction model from examples/friction_cal.py")
ap.add_argument("--fric-scale", type=float, default=0.8, help="fraction of calibrated friction to compensate")
ap.add_argument("--serve", nargs="?", const=8731, type=int, default=None, metavar="PORT",
                help="web panel on localhost (default port 8731): live friction-compensation slider")
a = ap.parse_args()

gc = GravityCompensator(can=a.can, payload_mass=a.payload_mass, tool=a.tool,
                        friction=a.friction, fric_scale=a.fric_scale)
if gc.fric is not None:
    print(gc.fric.summary())
if a.serve is not None:
    if gc.fric is None:
        ap.error("--serve needs --friction (the slider controls the friction compensation)")
    from piperx_teleop import webserve
    webserve.start(gc, os.path.join(os.path.dirname(os.path.abspath(__file__)), "drag_panel.html"), port=a.serve)
    print("panel: http://127.0.0.1:%d   (slider = friction compensation 0..0.9, live)" % a.serve)
print("pose (deg)   :", np.degrees(gc.q()).round(1))
print("gravity (N.m):", gc.gravity().round(2))
if not a.yes:
    input(">>> ENTER to go compliant - the arm will be freely draggable ")
trip = gc.run(duration=a.duration)
print("position hold restored" + (" (%s)" % trip if trip else ""))
