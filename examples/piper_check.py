"""Read-only Piper health check: CAN link, joint state, gripper, status, firmware.

Nothing moves. Exit code 0 means the arm answered on the bus.

    conda activate piperctl
    python examples/piper_check.py [can0]
"""
import subprocess
import sys
import time

import piper_sdk

CAN = sys.argv[1] if len(sys.argv) > 1 else "can0"

link = subprocess.run(["ip", "-br", "link", "show", CAN], capture_output=True, text=True)
if link.returncode != 0 or " UP " not in link.stdout:
    sys.exit(f"{CAN} is not up. Bring it up with:\n"
             f"    sudo ip link set {CAN} up type can bitrate 1000000")
print(f"{CAN}: up")

p = piper_sdk.C_PiperInterface_V2(can_name=CAN)
p.ConnectPort()
time.sleep(0.5)

j = p.GetArmJointMsgs().joint_state
q = [getattr(j, f"joint_{i}") / 1000 for i in range(1, 7)]
if all(v == 0 for v in q) and p.GetArmJointMsgs().time_stamp == 0:
    sys.exit("no joint feedback: bus is up but the arm is not answering "
             "(power? CAN adapter plugged in? correct interface?)")
print("joints (deg):", [round(v, 1) for v in q])
print("gripper (mm):", p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000)

s = p.GetArmStatus().arm_status
print(f"ctrl mode: {s.ctrl_mode}  arm status: {s.arm_status}  teach: {s.teach_status}  err: {s.err_code}")
if s.teach_status != 0:
    print("!! teach mode is active: commands will be silently discarded until "
          "you press the teach button on the arm")
if s.err_code != 0:
    print("!! error code nonzero, see err_status above")

try:
    print("firmware:", p.GetPiperFirmwareVersion())
except Exception as e:  # noqa: BLE001
    print("firmware read failed:", e)
print("OK")
