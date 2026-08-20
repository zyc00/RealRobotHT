"""Emergency release.

    python scripts/release.py            # limp: arm motors AND gripper de-energised
                                         #       THE ARM WILL FALL - support it first
    python scripts/release.py --hold     # freeze arm where it stands, torque on
    python scripts/release.py --open     # just open the gripper, arm untouched

DisableArm(7) only covers joints 1-6.  The gripper is a separate motor reached
through GripperCtrl, so a plain DisableArm leaves it clamped - hence the
explicit gripper handling here.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from piper_sdk import C_PiperInterface_V2

GRIPPER_DISABLE = 0x00
GRIPPER_ENABLE = 0x01
OPEN_M = 0.07

p = C_PiperInterface_V2("can0")
p.ConnectPort()
time.sleep(0.2)


def gripper_state():
    g = p.GetArmGripperMsgs().gripper_state
    return g.grippers_angle * 1e-6, g.grippers_effort * 1e-3


if "--open" in sys.argv:
    for _ in range(25):
        p.GripperCtrl(int(OPEN_M * 1e6), 1000, GRIPPER_ENABLE, 0)
        time.sleep(0.02)
    time.sleep(0.4)
    a, e = gripper_state()
    print("gripper opened -> %.1f mm, effort %.2f N.m" % (a * 1000, e))

elif "--hold" in sys.argv:
    j = p.GetArmJointMsgs().joint_state
    q = [j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]
    for _ in range(40):
        p.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        p.JointCtrl(*[int(v) for v in q])
        time.sleep(0.01)
    print("holding at", np.round(np.array(q) / 1000.0, 2), "deg")

else:
    a0, e0 = gripper_state()
    # Gripper first: de-energising it while the arm still holds pose means the
    # tool opens before the arm goes slack, rather than after it has dropped.
    for _ in range(15):
        p.GripperCtrl(0, 0, GRIPPER_DISABLE, 0)
        time.sleep(0.01)
    p.DisableArm(7)
    time.sleep(0.3)
    a1, e1 = gripper_state()
    print("motors OFF - arm is limp. enable status:", p.GetArmEnableStatus())
    print("gripper: %.1f mm -> %.1f mm, effort %.2f -> %.2f N.m" % (a0 * 1000, a1 * 1000, e0, e1))
