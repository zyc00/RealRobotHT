"""Replay a recorded trajectory and measure how faithfully it reproduces.

    python scripts/replay_teleop.py data/ep01.npz
    python scripts/replay_teleop.py data/ep01.npz --dry-run    # no motion

The recorded `action` rows are absolute Cartesian poses in the firmware's frame
and are already rate limited, so replaying them at the recorded rate should
reproduce the motion.  Two things make that less automatic than it sounds, and
both are handled here:

  * The firmware's IK picks a solution from the CURRENT configuration, so we
    home to the recorded starting joints first.  Starting elsewhere can resolve
    the same EE poses onto a different joint path.
  * A recorded stall replays as a stall.  Where the arm could not reach during
    recording, the action stopped advancing, so the replay reproduces that too.

Afterwards it reports position error against the recorded end-effector path.
"""
import argparse
import sys
import time

import numpy as np

from piperx_teleop import PiperArm
from piperx_teleop.arm import JOINT_LIMITS, MOVE_P

ap = argparse.ArgumentParser()
ap.add_argument("episode")
ap.add_argument("--can", default="can0")
ap.add_argument("--speed", type=int, default=20)
ap.add_argument("--rate-scale", type=float, default=1.0, help="<1 replays slower")
ap.add_argument("--clutched-only", action="store_true",
                help="replay only the demonstrating ticks")
ap.add_argument("--max-joint-step", type=float, default=3.0)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

d = np.load(a.episode)
act, q_rec, ee_rec = d["action"], d["q"], d["ee_pos"]
t_rec = d["t"] - d["t"][0]
if a.clutched_only:
    m = d["clutch"].astype(bool)
    act, q_rec, ee_rec, t_rec = act[m], q_rec[m], ee_rec[m], t_rec[m]
    t_rec = t_rec - t_rec[0]
n = len(act)
print("episode %s: %d ticks, %.1f s, %.0f Hz"
      % (a.episode, n, t_rec[-1], n / max(t_rec[-1], 1e-6)))
print("recorded EE span (mm):", ((ee_rec.max(0) - ee_rec.min(0)) * 1000).round(0))

arm = PiperArm(a.can).connect()
q0 = q_rec[0]
margin = np.degrees(np.minimum(q0 - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q0))
print("\nstart joints (deg):", np.degrees(q0).round(1), " margin:", margin.round(0))
if margin.min() < 2:
    sys.exit("recorded start pose is against a joint limit; refusing to replay")

if a.dry_run:
    print("\nDRY RUN - not moving. First/last commanded poses:")
    print("  ", act[0][:3].round(3), "->", act[-1][:3].round(3))
    sys.exit(0)

input("\n>>> Workspace clear? ENTER to home to the recorded start pose ")
if not arm.move_to(q0, speed_pct=15, timeout=25):
    sys.exit("could not reach the recorded start pose")
time.sleep(0.8)
p0, _ = arm.end_pose()
print("homed. EE at %s (recorded start %s)" % (p0.round(3), ee_rec[0].round(3)))

print("\nreplaying...")
achieved, q_ach = [], []
prev_q = arm.q()
aborted = None
t_start = time.time()
for i in range(n):
    pos, rpy, grip = act[i][:3], act[i][3:6], act[i][6]
    arm.end_pose_ctrl(pos, rpy, MOVE_P, a.speed)
    arm.set_gripper(grip > 0.5)

    # pace to the recorded timeline
    want = t_rec[i] / max(a.rate_scale, 1e-6)
    while time.time() - t_start < want:
        time.sleep(0.001)

    cur = arm.q()
    jump = np.degrees(np.abs(cur - prev_q)).max()
    prev_q = cur
    if jump > a.max_joint_step:
        aborted = "joint jumped %.1f deg at tick %d (IK branch flip?)" % (jump, i)
        break
    p, _ = arm.end_pose()
    achieved.append(p)
    q_ach.append(cur)
    if i % 100 == 0:
        sys.stdout.write("\r  %5d/%d  ee %s  " % (i, n, p.round(3)))
        sys.stdout.flush()

arm.hold()
achieved = np.array(achieved)
q_ach = np.array(q_ach)
k = len(achieved)
print("\n\nreplayed %d/%d ticks" % (k, n))
if aborted:
    print("  ABORTED: %s" % aborted)
if k > 10:
    err = np.linalg.norm(achieved - ee_rec[:k], axis=1) * 1000
    jerr = np.degrees(np.abs(q_ach - q_rec[:k])).max(1)
    print("  EE position error vs recording: mean %.1f mm  p95 %.1f mm  max %.1f mm"
          % (err.mean(), np.percentile(err, 95), err.max()))
    print("  worst joint error:              mean %.1f deg  max %.1f deg"
          % (jerr.mean(), jerr.max()))
    if err.mean() < 10:
        print("\n  the trajectory reproduces well")
    elif err.mean() < 30:
        print("\n  approximate - check whether the arm stalled during recording")
    else:
        print("\n  poor: likely a different IK branch, or the start pose drifted")
