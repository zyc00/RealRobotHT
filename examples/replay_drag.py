"""Replay a drag-recorded demonstration (from examples/record_drag.py).

Moves to the first recorded pose, waits for confirmation, then streams the
recorded joints at the recorded timing through the normal position loop.

    python examples/replay_drag.py data/demo_001.npz
"""
import sys
import time

import numpy as np
import piper_sdk

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/demo.npz"
CAN = sys.argv[2] if len(sys.argv) > 2 else "can0"
RAD = np.pi / 180.0

d = np.load(PATH)
T, Q = d["t"], d["q"]
print("%s: %d samples, %.1f s" % (PATH, len(T), T[-1]))

p = piper_sdk.C_PiperInterface_V2(can_name=CAN)
p.ConnectPort()
time.sleep(0.5)
if not all(p.GetArmEnableStatus()):
    p.EnableArm(7)
    time.sleep(1.5)


def cmd(q):
    return [int(round(v / RAD * 1000)) for v in q]


def q_now():
    j = p.GetArmJointMsgs().joint_state
    return np.array([j.joint_1, j.joint_2, j.joint_3,
                     j.joint_4, j.joint_5, j.joint_6]) * 1e-3 * RAD


input(">>> Workspace clear? ENTER to move to the start pose ")
p.MotionCtrl_2(0x01, 0x01, 15, 0x00, 0, 0x01)
t0 = time.time()
while time.time() - t0 < 15:
    p.JointCtrl(*cmd(Q[0]))
    if np.abs(q_now() - Q[0]).max() < np.radians(1.0):
        break
    time.sleep(0.01)

input(">>> At start. ENTER to replay ")
p.MotionCtrl_2(0x01, 0x01, 30, 0x00, 0, 0x01)
t0 = time.time()
for i in range(len(T)):
    while time.time() - t0 < T[i]:
        time.sleep(0.001)
    p.JointCtrl(*cmd(Q[i]))
err = np.degrees(np.abs(q_now() - Q[-1]).max())
print("done - final pose error %.2f deg" % err)
