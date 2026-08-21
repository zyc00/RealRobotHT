"""Record a demonstration by dragging the arm in the FIRMWARE's drag mode.

This is the usable path: MasterSlaveConfig(0xFA) engages the same
gravity-compensated drag as the teach button (the one that feels right), and
the arm broadcasts its pose as JointCtrl frames on 0x155-0x157 - literally the
commands a follower would execute, i.e. exactly the actions a policy should
learn. This script enters drag mode, records that stream, and restores
position hold.

    python examples/record_drag.py -o data/demo_001.npz
    python examples/replay_drag.py    data/demo_001.npz

Saved keys: t (s, from start), q (rad, 6), gripper_m (m, if the gripper
stream is alive), plus hz.
"""
import argparse
import time

import numpy as np
import piper_sdk

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="data/demo.npz")
ap.add_argument("--can", default="can0")
ap.add_argument("--hz", type=float, default=50.0)
a = ap.parse_args()

RAD = np.pi / 180.0
p = piper_sdk.C_PiperInterface_V2(can_name=a.can)
p.ConnectPort()
time.sleep(0.5)


def hold_here():
    p.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
    time.sleep(0.3)
    j = p.GetArmJointMsgs().joint_state
    p.MotionCtrl_2(0x01, 0x01, 20, 0x00, 0, 0x01)
    time.sleep(0.05)
    for _ in range(30):
        p.JointCtrl(j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6)
        time.sleep(0.01)


def leader_q():
    """The 0x155-0x157 broadcast: the pose as a follower-ready command, rad."""
    c = p.GetArmJointCtrl()
    jc = c.joint_ctrl
    return np.array([jc.joint_1, jc.joint_2, jc.joint_3,
                     jc.joint_4, jc.joint_5, jc.joint_6]) * 1e-3 * RAD, c.Hz


input(">>> ENTER to enter drag mode (arm holds itself, like the teach button) ")
p.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00)
time.sleep(0.3)
q, hz = leader_q()
if hz == 0.0:
    time.sleep(0.7)
    q, hz = leader_q()
print("leader stream: %.0f Hz" % hz)

input(">>> Move the arm to your START pose, then ENTER to begin recording ")
print("recording at %.0f Hz - Ctrl-C to stop\n" % a.hz)

T, Q, G = [], [], []
t0 = time.time()
try:
    while True:
        q, _ = leader_q()
        g = p.GetArmGripperMsgs().gripper_state.grippers_angle * 1e-6
        T.append(time.time() - t0)
        Q.append(q)
        G.append(g)
        if len(T) % int(a.hz * 2) == 0:
            print("  %6.1f s  q=%s" % (T[-1], np.degrees(q).round(1)))
        time.sleep(1.0 / a.hz)
except KeyboardInterrupt:
    pass
finally:
    hold_here()

T, Q, G = np.array(T), np.array(Q), np.array(G)
np.savez(a.output, t=T, q=Q, gripper_m=G, hz=a.hz)
print("\nsaved %d samples (%.1f s) -> %s" % (len(T), T[-1] if len(T) else 0, a.output))
print("arm is back in position hold.")
