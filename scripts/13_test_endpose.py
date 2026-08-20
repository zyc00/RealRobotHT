"""Guarded probe of the firmware's Cartesian control - no Quest needed.

Commands small relative steps from the firmware's OWN reported end pose and
watches the joints. A small Cartesian step must produce a small joint step; a
big one means the firmware picked a different IK branch or hit a singularity.
Aborts to joint hold the moment that happens.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.cartesian import CartesianController, MOVE_P, MOVE_L

STEP = 0.002          # metres per control cycle
N_STEPS = 25          # -> 50 mm total per leg
SPEED = 10
WATCHDOG_DEG = 14.0   # loosened for characterisation; we log the real distribution
# J5 deliberately off zero: the wrist singularity sits at J5=0 and is where
# Cartesian IK is worst conditioned.
START_Q = np.radians([0.0, 45.0, -80.0, 0.0, 30.0, 0.0])

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
JUMPS = []
arm = PiperArm().connect(0.5)
if not all(arm.is_enabled()):
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)

print("moving to a well-conditioned start pose", np.degrees(START_Q))
t0 = time.time()
while time.time() - t0 < 20:
    arm.move_j(START_Q, speed_pct=15)
    if np.abs(arm.q() - START_Q).max() < np.radians(1.0):
        break
    time.sleep(0.01)
time.sleep(1.0)
print("joints now:", np.degrees(arm.q()).round(2))


def leg(ctl, axis, sign, label):
    d = np.zeros(3); d[axis] = sign * STEP
    q0 = arm.q(); p0, _ = ctl.read_pose()
    worst = 0.0
    jumps = []
    qprev = arm.q()
    for _ in range(N_STEPS):
        if not ctl.check_joints():
            return False, ctl.aborted, worst
        ctl.apply_delta(d)
        time.sleep(0.02)
        qn = arm.q()
        jumps.append(np.degrees(np.abs(qn - qprev)).max()); qprev = qn
        worst = max(worst, np.degrees(np.abs(qn - q0)).max())
    JUMPS.extend(jumps)
    time.sleep(0.6)
    p1, _ = ctl.read_pose()
    moved = p1 - p0
    want = d * N_STEPS
    print("  %-10s commanded %s -> achieved %s  (%3.0f%%)  per-cycle joint jump: med %.1f p95 %.1f max %.1f deg"
          % (label, (want * 1000).round(1), (moved * 1000).round(1),
             100 * np.dot(moved, want) / max(np.dot(want, want), 1e-9),
             np.median(jumps), np.percentile(jumps, 95), np.max(jumps)))
    return True, None, worst


for mode, name in [(MOVE_P, "MOVE_P"), (MOVE_L, "MOVE_L")]:
    print("\n=== %s ===" % name)
    ctl = CartesianController(arm, max_step_m=STEP * 1.5, max_reach_m=0.12, min_z_m=0.10,
                              max_joint_step_deg=WATCHDOG_DEG, speed_pct=SPEED,
                              move_mode=mode)
    ctl.start()
    print("  start pose (m):", ctl.origin.round(4), " rot(deg):", ctl.rot.round(1))
    ok = True
    for axis, sign, label in [(0, +1, "+X fwd"), (0, -1, "-X back"),
                              (2, +1, "+Z up"), (2, -1, "-Z down"),
                              (1, +1, "+Y left"), (1, -1, "-Y right")]:
        ok, why, _ = leg(ctl, axis, sign, label)
        if not ok:
            print("  !! ABORT: %s" % why)
            ctl.hold_joints()
            break
    st = arm.piper.GetArmStatus().arm_status
    print("  motion_status=%s err=%s clamps=%d" % (st.motion_status, st.err_code, ctl.clamp_events))
    if not ok:
        break
    # return home between modes
    t0 = time.time()
    while time.time() - t0 < 15:
        arm.move_j(START_Q, speed_pct=15)
        if np.abs(arm.q() - START_Q).max() < np.radians(1.0):
            break
        time.sleep(0.01)
    time.sleep(0.5)

q = arm.q()
for _ in range(30):
    arm.move_j(q, speed_pct=10); time.sleep(0.01)
if JUMPS:
    J = np.array(JUMPS)
    print("\nper-cycle joint jump over all legs (deg): med %.2f  p95 %.2f  p99 %.2f  max %.2f  (n=%d)"
          % (np.median(J), np.percentile(J, 95), np.percentile(J, 99), J.max(), len(J)))
    print("suggested watchdog = %.1f deg (2x p99)" % (2 * np.percentile(J, 99)))
print("\ndone; arm holding at", np.degrees(arm.q()).round(2))
