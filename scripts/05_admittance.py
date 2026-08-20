"""Interactive admittance run - the arm follows your hand.

  python scripts/05_admittance.py                 # J2+J3, conservative
  python scripts/05_admittance.py --all           # all six joints
  python scripts/05_admittance.py --damping 8     # lighter = moves easier
  python scripts/05_admittance.py --pose 40 -90   # starting J2, J3 in degrees

Ctrl-C at any time to stop and hold.
"""
import argparse, sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm, JOINT_LIMITS
from piper_ht.model import PiperModel
from piper_ht.gravity_fit import GravityFit
from piper_ht.admittance import AdmittanceController

ap = argparse.ArgumentParser()
ap.add_argument("--only", type=int, nargs="+", metavar="J",
                help="enable only these joints, 1-indexed (default: all six)")
ap.add_argument("--damping", type=float, default=None, help="scale all damping (lower = lighter)")
ap.add_argument("--deadband", type=float, default=None, help="scale all deadbands")
ap.add_argument("--vmax", type=float, default=0.8)
ap.add_argument("--hyst", type=float, default=None, help="breakaway ratio, lower = lighter once moving")
ap.add_argument("--mass", type=float, default=0.0, help="virtual inertia; smooths velocity, costs response")
ap.add_argument("--light", action="store_true", help="preset: low damping + inertia smoothing")
ap.add_argument("--no-next", action="store_true", help="use the analytic gravity model instead of NEXT")
ap.add_argument("--pose", type=float, nargs=2, default=[40.0, -90.0], metavar=("J2", "J3"))
ap.add_argument("--speed", type=int, default=55)
ap.add_argument("--vmax-default", dest="_unused", help=argparse.SUPPRESS)
a = ap.parse_args()

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
mdl = PiperModel()
ctl = AdmittanceController(arm, GravityFit(model=mdl), v_max=a.vmax, speed_pct=a.speed)
if not a.no_next:
    try:
        from piper_ht.next_estimator import NextEstimator
        ctl.attach_next(NextEstimator(fallback=ctl.grav))
        print("using LEARNED (NEXT) free-space model")
    except Exception as e:
        print("NEXT unavailable (%s); falling back to the analytic model" % e)
if a.only:
    ctl.enabled[:] = False
    for j in a.only:
        ctl.enabled[j - 1] = True
if a.damping:
    ctl.damping = ctl.damping * 0 + a.damping
if a.hyst is not None:
    ctl.hysteresis = a.hyst
if a.light:
    ctl.damping = np.array([3.0, 3.5, 3.0, 3.0, 2.0, 1.5])
    ctl.hysteresis = 0.25
    ctl.mass = 0.35 * np.ones(6)
if a.mass:
    ctl.mass = a.mass * np.ones(6)
if a.deadband:
    ctl.deadband = ctl.deadband * a.deadband

TEST = np.radians([0.0, a.pose[0], a.pose[1], 0.0, 0.0, 0.0])
margin = np.degrees(np.minimum(TEST - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - TEST))
if margin.min() < 8:
    sys.exit("!! start pose is within %.1f deg of a joint limit; the end stop "
             "would absorb the load. Pick another --pose." % margin.min())

print("joints enabled :", ["J%d" % (i+1) for i in range(6) if ctl.enabled[i]])
print("deadband (N.m) :", ctl.deadband.round(2))
print("damping        :", ctl.damping.round(1))
print("virtual mass   :", ctl.mass.round(2), "(filter tau = %.2f s)" % (ctl.mass[1]/ctl.damping[1]))
print("v_max          : %.2f rad/s   leash %.0f deg   travel cap %.0f deg" % (
    ctl.v_max, np.degrees(ctl.leash), np.degrees(ctl.max_travel)))

if not all(arm.is_enabled()):
    print("\nenabling...")
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)

input("\n>>> Workspace clear? ENTER to move to the start pose ")
t0 = time.time()
while time.time() - t0 < 15:
    arm.move_j(TEST, speed_pct=15)
    if np.abs(arm.q() - TEST).max() < np.radians(0.8):
        break
    time.sleep(0.01)
time.sleep(0.8)
print("at:", np.degrees(arm.q()).round(2))

input("\n>>> HANDS OFF, then ENTER to capture the zero ")
z, n = ctl.capture_zero(2.0)
print("zero :", z.round(3))
print("noise:", n.round(4))

print("\n>>> PUSH THE ARM - it should now follow your hand. Ctrl-C to stop.\n")
print("%6s  %-42s %-30s %s" % ("t", "tau_user (N.m)", "qdot_cmd (rad/s)", "q (deg)"))
ctl.start()
log = []
t_prev = time.time(); t0 = t_prev
try:
    k = 0
    while True:
        now = time.time()
        dt = min(now - t_prev, 0.05); t_prev = now
        q, tau_user, qdot, stop = ctl.step(dt)
        log.append((now - t0, q.copy(), tau_user.copy(), qdot.copy()))
        if k % 40 == 0:
            print("%6.1f  %s  %s  %s" % (
                now - t0,
                " ".join("%6.2f" % v for v in tau_user),
                " ".join("%5.2f" % v for v in qdot),
                np.degrees(q).round(1)))
        if stop:
            print("\n!! stopping: %s" % stop); break
        k += 1
        time.sleep(0.005)
except KeyboardInterrupt:
    print("\ninterrupted")

q_end = arm.q()
for _ in range(40):
    arm.move_j(q_end, speed_pct=10); time.sleep(0.01)

T = np.array([r[0] for r in log]); Q = np.array([r[1] for r in log])
U = np.array([r[2] for r in log]); V = np.array([r[3] for r in log])
np.savez("data/admittance_run.npz", t=T, q=Q, tau_user=U, qdot=V, zero=ctl.zero)
print("\ncaptured %d samples over %.1f s" % (len(T), T[-1] if len(T) else 0))
print("total joint travel (deg):", np.degrees(Q.max(0) - Q.min(0)).round(1))
print("peak |tau_user| (N.m)   :", np.abs(U).max(0).round(2))
print("peak |qdot_cmd| (rad/s) :", np.abs(V).max(0).round(3))
print("saved data/admittance_run.npz")
