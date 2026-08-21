"""Re-assert the firmware gravity-model settings: upright mount + FULL end load.

The teach/drag compensation degraded twice, both times after sessions in which
our scripts streamed MotionCtrl_2 with installation_pos=0x00 ("Invalid"). All
call sites now send 0x01, and this script re-asserts both settings on demand.
Pins the target to the current pose, so nothing moves.

    python examples/assert_arm_config.py
"""
import sys
import time

import piper_sdk

CAN = sys.argv[1] if len(sys.argv) > 1 else "can0"
p = piper_sdk.C_PiperInterface_V2(can_name=CAN)
p.ConnectPort()
time.sleep(0.5)

p.ArmParamEnquiryAndConfig(0x00, 0x00, 0x00, 0xAE, 0x02)     # end load FULL
time.sleep(0.1)
j = p.GetArmJointMsgs().joint_state
p.MotionCtrl_2(0x01, 0x01, 10, 0x00, 0, 0x01)                # upright mount
p.JointCtrl(j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6)
print("asserted: installation_pos=0x01 (upright), end_load=0x02 (FULL)")
print("neither is readable back - verify with the teach probe:")
print("    press the teach button, then: python -u examples/probe_teach_mode.py 25")
