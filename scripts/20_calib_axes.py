"""Find how the FIRMWARE's Cartesian axes map to the physical world.

Everything upstream was verified in software, but nothing ever checked that
firmware +X actually points away from the operator.  This moves the arm along
each firmware axis and asks you which way it really went, then saves the
correction so teleop directions come out right.

Small moves (40 mm) from the home pose, returning after each.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.cartesian import CartesianController, MOVE_P

CHOICES = {
    "1": ("away from me (forward)", np.array([1.0, 0, 0])),
    "2": ("toward me (back)",       np.array([-1.0, 0, 0])),
    "3": ("to my left",             np.array([0, 1.0, 0])),
    "4": ("to my right",            np.array([0, -1.0, 0])),
    "5": ("up",                     np.array([0, 0, 1.0])),
    "6": ("down",                   np.array([0, 0, -1.0])),
    "0": ("barely moved / unsure",  None),
}

arm = PiperArm().connect(0.5)
if arm.in_teach_mode():
    sys.exit("arm is in TEACH MODE - exit it first")
HOME = np.load("data/home_pose.npz")["q"] if __import__("os").path.exists("data/home_pose.npz") \
    else np.radians([0, 50, -75, 0, 20, 0])


def goto(qd, t=20):
    t0 = time.time()
    while time.time() - t0 < t:
        arm.move_j(qd, speed_pct=15)
        if np.abs(arm.q() - qd).max() < np.radians(1.2):
            return True
        time.sleep(0.01)
    return False


print("Stand where you normally teleoperate, facing the arm as usual.")
input(">>> ENTER to home the arm ")
goto(HOME)
time.sleep(0.8)

M = np.zeros((3, 3))
for ax, name in enumerate(["firmware +X", "firmware +Y", "firmware +Z"]):
    goto(HOME); time.sleep(0.6)
    ctl = CartesianController(arm, max_step_m=0.004, max_reach_m=0.25, min_z_m=0.05,
                              max_joint_step_deg=2.0, speed_pct=15, move_mode=MOVE_P)
    ctl.start()
    d = np.zeros(3); d[ax] = 1.0
    print("\nmoving along %s by 40 mm - WATCH THE TOOL" % name)
    for k in range(40):
        if not ctl.check_joints():
            print("  (watchdog: %s)" % ctl.aborted); break
        ctl.follow(d * 0.040 * min(1.0, (k + 1) / 35))
        time.sleep(0.03)
    time.sleep(0.6)
    _, act = ctl.tracking_error()
    moved = np.linalg.norm(act - ctl.origin) * 1000
    print("  (arm reports it moved %.0f mm)" % moved)
    for k, (label, _) in CHOICES.items():
        print("     %s = %s" % (k, label))
    while True:
        c = input("  which way did the TOOL actually go? ").strip()
        if c in CHOICES:
            break
    label, vec = CHOICES[c]
    if vec is None:
        print("  skipped"); continue
    M[ax] = vec
    print("  -> %s means %s" % (name, label))
    goto(HOME)

goto(HOME)
q = arm.q()
for _ in range(25):
    arm.move_j(q, speed_pct=10); time.sleep(0.01)

print("\nfirmware axis -> physical direction:")
names = ["fwd", "left", "up"]
for i, n in enumerate(["+X", "+Y", "+Z"]):
    v = M[i]
    if not v.any():
        print("  %s : (not measured)" % n); continue
    j = int(np.argmax(np.abs(v)))
    print("  %s -> %s%s" % (n, "+" if v[j] > 0 else "-", names[j]))

if np.linalg.matrix_rank(M) == 3:
    np.savez("data/axis_calib.npz", M=M)
    print("\nsaved data/axis_calib.npz")
    if np.allclose(M, np.eye(3)):
        print("firmware axes already match the assumed convention - no correction needed.")
    else:
        print("correction WILL be applied by the teleop script.")
else:
    print("\nincomplete - re-run and answer all three.")
