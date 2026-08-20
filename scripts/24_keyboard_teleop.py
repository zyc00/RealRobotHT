"""Keyboard Cartesian teleop - no VR, no orientation. Position only.

A deliberately minimal path to the arm: keys -> Cartesian target -> firmware.
Orientation is held fixed at whatever it was when you started.

POSITION
  arrow up / down     z  up / down
  arrow left / right  y  left / right   (+y is the robot's left)
  w / s               x  forward / back
  [ / ]               smaller / larger position step

ORIENTATION  (rotations about the robot base axes, same axes as above)
  u / p               roll   about x (the forward axis)
  i / k               pitch  about y (tilt the tool down / up)
  j / l               yaw    about z (turn the tool left / right)
  - / =               smaller / larger rotation step
  r                   reset orientation back to the start attitude

  space               stop and re-anchor here (position AND orientation)
  h                   go home
  g                   toggle gripper
  o                   open gripper
  q or Ctrl-C         quit (arm holds position)

Key auto-repeat gives continuous motion when you hold a key.
"""
import argparse, os, select, sys, termios, time, tty
import numpy as np
sys.path.insert(0, ".")
from piper_ht.config import load as _load_config, describe as _cfg_path
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.cartesian import CartesianController, MOVE_P, MOVE_L

ap = argparse.ArgumentParser()
ap.add_argument("--step", type=float, default=0.005, help="metres per keypress")
ap.add_argument("--rot-step", type=float, default=3.0, help="degrees per keypress")
ap.add_argument("--max-rot-step", type=float, default=1.5,
                help="degrees of tool rotation per control cycle")
ap.add_argument("--lock-rotation", action="store_true", help="position only")
ap.add_argument("--speed", type=int, default=20)
ap.add_argument("--max-step", type=float, default=0.004, help="metres per control cycle")
ap.add_argument("--max-reach", type=float, default=0.25)
ap.add_argument("--min-z", type=float, default=0.05)
ap.add_argument("--movel", action="store_true")
ap.add_argument("--no-home", action="store_true")
_cfg = _load_config(keys=set(vars(ap.parse_known_args()[0]).keys()))
if _cfg:
    ap.set_defaults(**_cfg)
a = ap.parse_args()

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
if arm.in_teach_mode():
    sys.exit("arm is in TEACH MODE - press the teach button on the arm, then re-run")
if not all(arm.is_enabled()):
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)

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


if not a.no_home:
    print("homing to", np.degrees(HOME).round(1))
    goto(HOME); time.sleep(0.7)

ctl = CartesianController(arm, max_step_m=a.max_step, max_reach_m=a.max_reach,
                          min_z_m=a.min_z, max_joint_step_deg=2.0, speed_pct=a.speed,
                          move_mode=MOVE_L if a.movel else MOVE_P,
                          lock_rotation=a.lock_rotation, max_rot_step_deg=a.max_rot_step)
ctl.start()
disp = np.zeros(3)
# Accumulated orientation offset from the start attitude, kept as a rotation
# rather than a summed rotation vector: rotation vectors only add correctly for
# rotations about a common axis, and roll-then-pitch is not that.
R_disp = Rot.identity()
step = a.step
rot_step = a.rot_step
grip_closed = False
print(__doc__)
print("start pose: %s  rpy %s   step %.0f mm / %.1f deg\n"
      % (ctl.origin.round(3), np.round(ctl.rot, 1), step * 1000, rot_step))
if a.lock_rotation:
    print("rotation LOCKED (--lock-rotation)\n")

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)


ARROWS = {"[A": "up", "[B": "down", "[C": "right", "[D": "left",
          "OA": "up", "OB": "down", "OC": "right", "OD": "left"}


def getkey():
    """Read one key. Arrow keys arrive as ESC [ A/B/C/D across several reads.

    The trailing bytes can lag the ESC by a few milliseconds, so wait properly
    for them - a 1 ms timeout here made every arrow key look like a bare ESC.
    """
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    c = sys.stdin.read(1)
    if c != "\x1b":
        return c
    seq = ""
    t0 = time.time()
    while len(seq) < 2 and time.time() - t0 < 0.05:
        if select.select([sys.stdin], [], [], 0.01)[0]:
            seq += sys.stdin.read(1)
    return ARROWS.get(seq, "esc")


try:
    tty.setcbreak(fd)
    last = time.time()
    while True:
        k = getkey()
        if k in ("q", "\x03"):
            break
        elif k == "up":      disp[2] += step
        elif k == "down":    disp[2] -= step
        elif k == "left":    disp[1] += step
        elif k == "right":   disp[1] -= step
        elif k == "w":       disp[0] += step
        elif k == "s":       disp[0] -= step
        elif k == "[":       step = max(0.001, step / 1.5)
        elif k == "]":       step = min(0.05, step * 1.5)
        elif k in ("u", "p", "i", "k", "j", "l"):
            axis = {"u": (0, +1), "p": (0, -1), "i": (1, +1),
                    "k": (1, -1), "j": (2, +1), "l": (2, -1)}[k]
            v = np.zeros(3); v[axis[0]] = axis[1] * np.radians(rot_step)
            R_disp = Rot.from_rotvec(v) * R_disp
        elif k == "-":       rot_step = max(0.25, rot_step / 1.5)
        elif k == "=":       rot_step = min(20.0, rot_step * 1.5)
        elif k == "r":       R_disp = Rot.identity()
        elif k == " ":
            ctl.relatch(); disp[:] = 0; R_disp = Rot.identity()
        elif k == "h":
            goto(HOME); time.sleep(0.5); ctl.start(); disp[:] = 0; R_disp = Rot.identity()
        elif k == "g":
            grip_closed = not grip_closed; ctl.set_gripper(grip_closed)
        elif k == "o":
            grip_closed = False; ctl.set_gripper(False)

        if not ctl.check_joints():
            sys.stdout.write("\r\n!! %s\r\n" % ctl.aborted)
            ctl.hold_joints()
            break
        ctl.follow(disp, R_disp.as_rotvec())
        while ctl.notes:
            sys.stdout.write("\r\n!! %s\r\n" % ctl.notes.pop(0))

        if time.time() - last > 0.15:
            last = time.time()
            lag, act = ctl.tracking_error()
            rmag = np.degrees(np.linalg.norm(R_disp.as_rotvec()))
            sys.stdout.write("\r  pos %s | rpy %s | lag %3.0f mm | rot %+5.1f deg | "
                             "step %2.0f mm / %.1f deg | grip %s   "
                             % (act.round(3), np.round(ctl.rot, 1), lag * 1000, rmag,
                                step * 1000, rot_step,
                                "closed" if grip_closed else "open"))
            sys.stdout.flush()
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    q = arm.q()
    for _ in range(30):
        arm.move_j(q, speed_pct=10); time.sleep(0.01)
    print("\nstopped; arm holding position")
