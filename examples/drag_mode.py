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
ap.add_argument("--gravity", default=None, help="per-joint gravity correction from examples/gravity_cal.py")
ap.add_argument("--fric-scale", type=float, default=0.8, help="fraction of calibrated friction to compensate")
ap.add_argument("--serve", nargs="?", const=8731, type=int, default=None, metavar="PORT",
                help="web panel on localhost (default port 8731): live sliders")
ap.add_argument("--balance", type=float, default=None, metavar="KAPPA",
                help="inertia shaping (b601 balanced drag): max lightening ratio minus one, 0..2; 0 = observe only")
ap.add_argument("--balance-md", type=float, default=1.0, help="virtual mass at the tool centre, kg")
ap.add_argument("--balance-irot", type=float, default=0.02, help="virtual rotational inertia about the tool x/y, kg.m^2")
ap.add_argument("--balance-damp", type=float, nargs=2, default=(0.0, 0.0), metavar=("T", "R"),
                help="saturating Cartesian damping (N.s/m, N.m.s/rad), the passivity guard; 0 0 = off")
ap.add_argument("--tcp", type=float, default=0.19, help="tool centre along the flange z, m (fingertips ~0.19)")
ap.add_argument("--damp", type=float, nargs=2, default=None, metavar=("T", "R"),
                help="passivity-aware Cartesian damping WITHOUT shaping (N.s/m, N.m.s/rad), paper eq 35")
ap.add_argument("--damp-vsat", type=float, default=0.15, help="damping saturation knee, rad/s")
ap.add_argument("--fric-sigma-v", type=float, default=0.0, help="friction schedule eq 36: velocity knee rad/s (0 = off)")
ap.add_argument("--fric-kappa0", type=float, default=0.0, help="friction schedule eq 37: cond(J) reference (0 = off)")
a = ap.parse_args()

bal = damping = dyn = None
if a.balance is not None or a.damp is not None or a.fric_kappa0 > 0:
    from piperx_teleop import BalancedDrag, CartesianDamping
    from piperx_teleop.dynamics import ArmDynamics
    dyn = ArmDynamics(tool=a.tool, tcp_offset=a.tcp, gravity=a.gravity)
if a.balance is not None:
    d_t, d_r = a.damp if a.damp is not None else a.balance_damp
    bal = BalancedDrag(dyn, kappa=a.balance, m_d=a.balance_md, i_rot=a.balance_irot,
                       damp_t=d_t, damp_r=d_r, damp_vsat=a.damp_vsat)
elif a.damp is not None or a.fric_kappa0 > 0:
    d_t, d_r = a.damp if a.damp is not None else (0.0, 0.0)
    damping = CartesianDamping(dyn, d_t, d_r, a.damp_vsat)
gc = GravityCompensator(can=a.can, payload_mass=a.payload_mass, tool=a.tool,
                        friction=a.friction, fric_scale=a.fric_scale, balance=bal, gravity=a.gravity,
                        damping=damping, fric_sigma_v=a.fric_sigma_v, fric_kappa0=a.fric_kappa0)
if gc.damping is not None and gc.fric is not None and (gc.damping.d_t > 0 or gc.damping.d_r > 0):
    import numpy as _np
    q0 = gc.q(); Jd = dyn.jacobian(q0, "local"); Md = dyn.mass_matrix(q0)
    have = gc.damping.joint_damping(Jd, Md, 1.0 / gc.hz)
    need = CartesianDamping.required_joint_damping(gc.fric.delta_f(), gc.fric.v0)
    print("passivity bound (eq 35) at the start pose, joint damping N.m.s/rad:")
    print("   applied  ", have.round(3)); print("   required ", need.round(3), " (delta_f / v0)")
    short = _np.where(have < need)[0]
    print("   " + ("OK on every joint" if len(short) == 0 else "SHORT on J%s - raise --damp or lower --fric-scale" % ",".join(str(j + 1) for j in short)))
if gc.fric is not None and (a.fric_sigma_v > 0 or a.fric_kappa0 > 0):
    print("friction schedules: sigma_v %.3f rad/s (eq 36), kappa0 %.0f (eq 37)" % (a.fric_sigma_v, a.fric_kappa0))
if a.gravity:
    print("gravity correction: scale %s  bias %s N.m" % (gc.mdl.g_scale.round(3), gc.mdl.g_bias.round(3)))
if gc.fric is not None:
    print(gc.fric.summary())
if bal is not None:
    print("balanced drag: kappa %.2f (arm up to %.1fx lighter), virtual mass %.2f kg, i_rot %.3f, "
          "damping %s, tcp %.2f m%s" % (bal.kappa, 1 + bal.kappa, bal.lam_d[0], bal.lam_d[3],
                                        a.balance_damp, a.tcp, "  [OBSERVE ONLY]" if bal.kappa == 0 else ""))
if a.serve is not None:
    if gc.fric is None and bal is None and damping is None:
        ap.error("--serve needs --friction, --damp and/or --balance (the panel controls those layers)")
    from piperx_teleop import webserve
    webserve.start(gc, os.path.join(os.path.dirname(os.path.abspath(__file__)), "drag_panel.html"), port=a.serve)
    print("panel: http://127.0.0.1:%d   (live: friction 0..0.9, shaping kappa 0..2, virtual mass)" % a.serve)
print("pose (deg)   :", np.degrees(gc.q()).round(1))
print("gravity (N.m):", gc.gravity().round(2))
if not a.yes:
    input(">>> ENTER to go compliant - the arm will be freely draggable ")
trip = gc.run(duration=a.duration)
print("position hold restored" + (" (%s)" % trip if trip else ""))
