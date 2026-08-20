"""End-to-end direction check: your hand vs where the ARM ACTUALLY WENT.

Every earlier check stopped at an intermediate quantity.  This one records the
controller displacement AND the arm's own measured displacement, and reports
both in physical terms, using the firmware axis meanings you confirmed by eye
(+X away from you, +Y your left, +Z up).

The arm MOVES during this test.
"""
import sys, time, os
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.cartesian import CartesianController, MOVE_P
from piper_ht.quest_teleop import QuestCartesianSource, QuestTeleopConfig

GAIN = 0.5
TESTS = [("push your hand AWAY from your chest", "away from you"),
         ("lift your hand UP", "up"),
         ("move your hand to YOUR LEFT", "to your left")]
ROBOT = [("away from you", "toward you"), ("to your left", "to your right"), ("up", "down")]

cfg = QuestTeleopConfig()
if os.path.exists("data/yaw_calib.npz"):
    h = float(np.load("data/yaw_calib.npz")["heading_deg"])
    cfg.head_yaw_align = False; cfg.head_relative = False; cfg.frame_mode = "latched"
else:
    h = None
src = QuestCartesianSource(cfg).start()
if h is not None:
    src.frame.set_yaw_offset(h)
    print("using measured body frame, heading %.1f deg" % h)

arm = PiperArm().connect(0.5)
if arm.in_teach_mode():
    sys.exit("arm is in TEACH MODE")
HOME = np.load("data/home_pose.npz")["q"] if os.path.exists("data/home_pose.npz") \
    else np.radians([0, 50, -75, 0, 20, 0])

def goto(qd, t=20):
    t0 = time.time()
    while time.time() - t0 < t:
        arm.move_j(qd, speed_pct=15)
        if np.abs(arm.q() - qd).max() < np.radians(1.2):
            return True
        time.sleep(0.01)
    return False

print("waiting for headset...")
for _ in range(200):
    if src.poll().connected: break
    time.sleep(0.1)
if not src.poll().connected: sys.exit("Quest not connected")

def describe(v, names):
    j = int(np.argmax(np.abs(v)))
    if np.abs(v).max() < 0.008:
        return "(barely moved)"
    return names[j][0 if v[j] > 0 else 1]

for prompt, expected in TESTS:
    goto(HOME); time.sleep(0.8)
    ctl = CartesianController(arm, max_step_m=0.004, max_reach_m=0.25, min_z_m=0.05,
                              max_joint_step_deg=2.0, speed_pct=15, move_mode=MOVE_P,
                              position_gain=GAIN)
    ctl.start()
    p0 = ctl.origin.copy()
    input("\n>>> ENTER, then SQUEEZE and %s (~25 cm), hold, release. " % prompt.upper())
    world = np.zeros(3); seen = False
    t0 = time.time()
    while time.time() - t0 < 15:
        s = src.poll()
        if s.clutch:
            seen = True
            world += s.dpos_world
            if not ctl.check_joints():
                print("   watchdog: %s" % ctl.aborted); break
            ctl.follow(s.disp_pos)
        elif seen and np.linalg.norm(world) > 0.06:
            break
        time.sleep(0.008)
    time.sleep(0.7)
    _, act = ctl.tracking_error()
    arm_d = act - p0
    print("   you moved      : %s   (%.0f mm)" % (prompt.split()[-3:][0], np.linalg.norm(world) * 1000))
    print("   ARM moved      : %-18s (%.0f mm)  raw xyz %s mm"
          % (describe(arm_d, ROBOT), np.linalg.norm(arm_d) * 1000, (arm_d * 1000).round(0)))
    print("   expected       : %s" % expected)
    print("   %s" % ("OK" if describe(arm_d, ROBOT) == expected else "*** MISMATCH ***"))

goto(HOME)
q = arm.q()
for _ in range(25):
    arm.move_j(q, speed_pct=10); time.sleep(0.01)
src.stop()
print("\ndone")
