"""AgileX's OFFICIAL zero-force drag: firmware leader mode, no software loop.

This is how pyAgxArm implements gravity-compensated dragging - a single mode
switch, the same compensator that runs behind the teach button:

    set_leader_mode()  ==  MasterSlaveConfig(linkage_config=0xFA)

In leader mode the arm STOPS sending normal feedback (0x2Ax goes quiet - in
August we misread that as the arm dying) and instead BROADCASTS its joint
positions as JointCtrl command frames on 0x155/0x156/0x157, pre-formatted for a
follower arm.  This script decodes those to show the drag is streaming.

Exit restores 0xFC ("motion output arm"), which is what brought CAN back last
time, then re-arms position hold.

    python examples/leader_drag.py        # either env; no MIT patch involved

The arm should hold itself (teach-mode-grade compensation, ~60% gravity), but
keep a hand near it and `re` ready the first time.
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

j = p.GetArmJointMsgs().joint_state
q0 = np.array([j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]) * 1e-3
print("current pose (deg):", q0.round(1))
print("firmware:", p.GetPiperFirmwareVersion())

input("\n>>> Entering FIRMWARE ZERO-FORCE DRAG (leader mode). The arm should\n"
      ">>> hold itself like teach mode. Hand near it? ENTER to switch ")

p.MasterSlaveConfig(0xFA, 0x00, 0x00, 0x00)
print("\nleader mode set. Drag the arm - its pose now streams as JointCtrl")
print("frames on 0x155-0x157 (normal feedback is intentionally silent).")
print("Ctrl-C to exit and restore normal control.\n")

last = None
try:
    while True:
        ctrl = p.GetArmJointCtrl()
        jc = ctrl.joint_ctrl
        q = np.array([jc.joint_1, jc.joint_2, jc.joint_3,
                      jc.joint_4, jc.joint_5, jc.joint_6]) * 1e-3
        line = "0x155 stream: %s deg   (%.0f Hz)" % (q.round(1), ctrl.Hz)
        if line != last:
            print(line)
            last = line
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    print("\nrestoring: 0xFC, then position hold at the current pose...")
    p.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
    time.sleep(0.3)
    j = p.GetArmJointMsgs().joint_state
    p.MotionCtrl_2(0x01, 0x01, 20, 0x00, 0, 0x01)
    time.sleep(0.05)
    for _ in range(30):
        p.JointCtrl(j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6)
        time.sleep(0.01)
    jn = p.GetArmJointMsgs().joint_state
    print("holding at (deg):", (np.array([jn.joint_1, jn.joint_2, jn.joint_3,
          jn.joint_4, jn.joint_5, jn.joint_6]) * 1e-3).round(1))
    print("if feedback looks dead, power-cycle the arm - vendor docs allow for that.")
