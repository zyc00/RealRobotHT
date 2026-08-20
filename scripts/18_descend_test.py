"""Which start pose lets you actually move DOWN under fixed orientation?

EndPoseCtrl always takes a full 6-DOF pose, so translating holds the tool
orientation fixed - and the volume reachable under a fixed orientation is far
smaller than the arm's raw workspace.  This measures, for several candidate home
poses, how far the tool can descend before the firmware stops following.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.cartesian import CartesianController, MOVE_P

STEP = 0.002
MAX_STEPS = 90          # up to 180 mm
LAG_FAIL = 0.025        # arm considered stuck when this far behind

CANDIDATES = [
    ("home (J5=30)",      [0, 45, -80, 0, 30, 0]),
    ("tool down (J5=-40)", [0, 45, -80, 0, -40, 0]),
    ("elbow high",        [0, 30, -60, 0, 20, 0]),
    ("reach out",         [0, 60, -100, 0, 40, 0]),
    ("wrist neutral",     [0, 50, -90, 0, 0, 0]),
]

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
if not all(arm.is_enabled()):
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)


def goto(qd, timeout=18):
    t0 = time.time()
    while time.time() - t0 < timeout:
        arm.move_j(qd, speed_pct=15)
        if np.abs(arm.q() - qd).max() < np.radians(1.0):
            return True
        time.sleep(0.01)
    return False


print("%-20s %10s %10s  %s" % ("start pose", "descended", "limit", "stopped by"))
results = []
for name, degs in CANDIDATES:
    qd = np.radians(degs)
    if not goto(qd):
        print("%-20s        n/a        n/a  could not reach start pose" % name)
        continue
    time.sleep(0.9)
    ctl = CartesianController(arm, max_step_m=STEP * 1.5, max_reach_m=0.40,
                              min_z_m=0.03, max_joint_step_deg=2.0,
                              speed_pct=15, move_mode=MOVE_P)
    ctl.start()
    z0 = ctl.origin[2]
    why = "reached %d mm limit" % int(STEP * MAX_STEPS * 1000)
    got = 0.0
    for k in range(MAX_STEPS):
        if not ctl.check_joints():
            why = "joint watchdog (%s)" % ctl.aborted.split(" - ")[0]
            break
        ctl.apply_delta(np.array([0.0, 0.0, -STEP]))
        time.sleep(0.03)
        lag, actual = ctl.tracking_error()
        got = z0 - actual[2]
        if lag > LAG_FAIL:
            why = "arm stopped following (lag %.0f mm)" % (lag * 1000)
            break
    time.sleep(0.4)
    _, actual = ctl.tracking_error()
    got = z0 - actual[2]
    results.append((name, got, actual[2], why))
    print("%-20s %8.0f mm %8.3f m  %s" % (name, got * 1000, actual[2], why))
    # park high again before the next candidate
    goto(np.radians([0, 45, -80, 0, 30, 0]))

if results:
    best = max(results, key=lambda r: r[1])
    print("\nbest start pose: '%s' descended %.0f mm to z=%.3f m" % (best[0], best[1] * 1000, best[2]))
q = arm.q()
for _ in range(30):
    arm.move_j(q, speed_pct=10); time.sleep(0.01)
