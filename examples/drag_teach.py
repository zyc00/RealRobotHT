"""Second protocol door into the firmware's drag mode: MotionCtrl_1 teach-record.

The button's teach mode cannot be entered by protocol (AgileX: the button has
top priority in firmware, and ctrl_mode=0x02 was removed from the SDK as
invalid). But the drag-TEACH-RECORD command on CAN 0x150 is a supported way in:

    MotionCtrl_1(0x00, 0x00, 0x01)   # start teach recording -> drag mode
    MotionCtrl_1(0x00, 0x00, 0x02)   # end recording -> exits drag mode

It engages the same compensator the button does (the firmware records a
trajectory as a side effect; we simply never replay it). Compare the feel
against the teach button and against MasterSlaveConfig(0xFA).

    python examples/drag_teach.py
"""
import sys
import time

import numpy as np
import piper_sdk

CAN = sys.argv[1] if len(sys.argv) > 1 else "can0"
RAD = np.pi / 180.0

p = piper_sdk.C_PiperInterface_V2(can_name=CAN)
p.ConnectPort()
time.sleep(0.5)


def q_now():
    j = p.GetArmJointMsgs().joint_state
    return np.array([j.joint_1, j.joint_2, j.joint_3,
                     j.joint_4, j.joint_5, j.joint_6]) * 1e-3


print("pose (deg):", q_now().round(1))
input("\n>>> Entering drag-teach mode via MotionCtrl_1 (0x150). The arm should\n"
      ">>> hold itself like the teach button. Hand near it. ENTER ")

p.MotionCtrl_1(0x00, 0x00, 0x01)
time.sleep(0.3)
print("control mode now:", p.GetArmStatus().arm_status.ctrl_mode,
      "| teach state:", p.GetArmStatus().arm_status.teach_status)
print("\nDrag it. Ctrl-C to exit and restore position hold.\n")
try:
    while True:
        print("q (deg):", q_now().round(1), "   ", end="\r")
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    print("\nexiting drag-teach...")
    p.MotionCtrl_1(0x00, 0x00, 0x02)
    time.sleep(0.3)
    j = p.GetArmJointMsgs().joint_state
    p.MotionCtrl_2(0x01, 0x01, 20, 0x00, 0, 0x01)
    time.sleep(0.05)
    for _ in range(30):
        p.JointCtrl(j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6)
        time.sleep(0.01)
    print("position hold restored at", q_now().round(1))
