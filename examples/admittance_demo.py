"""DEMO: admittance control - push the arm and it follows your hand.

Kept as a demonstration, not a working teleop mode. It detects your pushes well
(SNR 47-135 on J2/J3), but it was never good enough to use:

  * breakaway force ~4.2 N against ~1.3 N of real joint friction, because
    admittance must INFER a push from torque and cannot tell your hand from the
    joint sitting somewhere in its friction band
  * the deadband is floored by gravity-model error, and the vendor URDF does not
    match this arm (51 mm rigid-fit residual), so the model cannot be made
    accurate enough
  * it drifts; the zeroing, hysteresis and drift guards below reduce it but do
    not remove it

UPDATE 2026-08-20: with end_load=FULL + installation_pos=0x01 loaded (power
cycle required), the FIRMWARE's own teach/drag mode now compensates properly
(J2 slope 0.76, gripper mass modeled) - for a leader arm, use that instead
(examples/leader_drag.py). This script asserts the same firmware settings so
the two feel consistent, and stays as the position-mode fallback.

The root cause is unfixable in software: MIT torque control is inert on firmware
S-V1.9-0, so real gravity compensation is impossible and this was the
workaround. VR teleop replaced the need for it.

Run it from a pose with room to move, keep a hand near Ctrl-C.

Original docstring follows.
--------------------------------------------------------------------
Interactive admittance run - the arm follows your hand.

  python examples/admittance_demo.py                # tuning from config/admittance.toml
  python examples/admittance_demo.py --only 2 3     # only J2 and J3 respond
  python examples/admittance_demo.py --damping 2    # override every joint's D
  python examples/admittance_demo.py --pose 40 -90  # starting J2, J3 in degrees

Everything tunable lives in config/admittance.toml; edit that rather than
hunting for constants in the source. CLI flags win over the file.

Ctrl-C at any time to stop and hold.
"""
import argparse, os, sys, time, tomllib
import numpy as np
sys.path.insert(0, ".")
from piperx_teleop.arm import JOINT_LIMITS
from piperx_teleop import PiperArm
from piper_ht.model import PiperModel
from piper_ht.gravity_fit import UNFITTED_DEADBAND, GravityFit
from piper_ht.admittance import AdmittanceController

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/admittance.toml")
ap.add_argument("--only", type=int, nargs="+", metavar="J",
                help="enable only these joints, 1-indexed (default: all six)")
ap.add_argument("--damping", type=float, default=None, help="scale all damping (lower = lighter)")
ap.add_argument("--deadband", type=float, default=None, help="scale all deadbands")
ap.add_argument("--vmax", type=float, default=None)
ap.add_argument("--hyst", type=float, default=None, help="breakaway ratio, lower = lighter once moving")
ap.add_argument("--mass", type=float, default=0.0, help="virtual inertia; smooths velocity, costs response")
ap.add_argument("--light", action="store_true",
                help="halve the config damping again and add inertia smoothing")
ap.add_argument("--no-next", action="store_true", help="use the analytic gravity model instead of NEXT")
ap.add_argument("--pose", type=float, nargs=2, default=None, metavar=("J2", "J3"))
ap.add_argument("--speed", type=int, default=None)
ap.add_argument("--vmax-default", dest="_unused", help=argparse.SUPPRESS)
a = ap.parse_args()

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

# Tuning comes from the file; the CLI is for one-off experiments on top of it.
cfg = {}
if os.path.exists(a.config):
    with open(a.config, "rb") as f:
        cfg = tomllib.load(f)
    print("config: %s" % a.config)
else:
    print("config: %s not found - using the controller's built-in defaults" % a.config)
control = cfg.get("control", {})
safety = cfg.get("safety", {})

arm = PiperArm().connect(0.5)

# Match the firmware state that fixed teach-mode gravity comp (2026-08-20):
# upright mount + FULL end load (empirically ~0.44 kg = the bare gripper).
# Both are transmit-only and only LOAD ON A POWER CYCLE, so this asserts the
# expected state rather than switching anything live. installation_pos rides
# on MotionCtrl_2, which re-arms MOVE J - pin the target to the current pose
# in the same breath so a stale target cannot move the arm.
arm.piper.ArmParamEnquiryAndConfig(0x00, 0x00, 0x00, 0xAE, 0x02)
_q = arm.q()
arm.piper.MotionCtrl_2(0x01, 0x01, 10, 0x00, 0, 0x01)
arm.move_j(_q, speed_pct=10)

mdl = PiperModel()
grav = GravityFit(model=mdl)
ctl = AdmittanceController(
    arm, grav,
    v_max=safety.get("v_max", 0.8),
    leash_deg=safety.get("leash_deg", 6.0),
    max_travel_deg=safety.get("max_travel_deg", 90.0),
    speed_pct=safety.get("speed_pct", 55),
)
if "damping" in control:
    d = np.asarray(control["damping"], float)
    if d.size != 6:
        sys.exit("!! config [control].damping needs 6 values, got %d" % d.size)
    if (d <= 0).any():
        sys.exit("!! config [control].damping must be positive; qdot = tau/D")
    ctl.damping = d
if "hysteresis" in control:
    ctl.hysteresis = np.asarray(control["hysteresis"], float) * np.ones(6)
if "mass" in control:
    ctl.mass = np.asarray(control["mass"], float) * np.ones(6)
if cfg.get("joints", {}).get("enabled") is not None:
    ctl.enabled = np.array(cfg["joints"]["enabled"], bool)
if not grav.fitted:
    ctl.deadband = UNFITTED_DEADBAND.copy()
    print("no identification sweep found (data/grav_fit.npz) - using the analytic\n"
          "URDF gravity model with wider deadbands. Expect more drift than the\n"
          "fitted model gives.\n")
if not cfg.get("model", {}).get("use_next", True):
    a.no_next = True
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
    ctl.hysteresis = a.hyst * np.ones(6)
if a.light:
    # Relative to whatever the config asked for, so the two compose.
    ctl.damping = ctl.damping / 2.0
    ctl.hysteresis = 0.25
    ctl.mass = 0.35 * np.ones(6)
if a.mass:
    ctl.mass = a.mass * np.ones(6)
ctl.deadband = ctl.deadband * float(control.get("deadband_scale", 1.0))
if a.deadband:
    ctl.deadband = ctl.deadband * a.deadband
if a.vmax is not None:
    ctl.v_max = a.vmax
if a.speed is not None:
    ctl.speed_pct = a.speed

pose = a.pose or cfg.get("start", {}).get("pose", [40.0, -90.0])
TEST = np.radians([0.0, pose[0], pose[1], 0.0, 0.0, 0.0])
margin = np.degrees(np.minimum(TEST - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - TEST))
if margin.min() < 8:
    sys.exit("!! start pose is within %.1f deg of a joint limit; the end stop "
             "would absorb the load. Pick another --pose." % margin.min())

print("joints enabled :", ["J%d" % (i+1) for i in range(6) if ctl.enabled[i]])
print("deadband (N.m) :", ctl.deadband.round(2))
print("damping        :", ctl.damping.round(1))
print("virtual mass   :", ctl.mass.round(2), "(filter tau = %.2f s)" % (ctl.mass[1]/ctl.damping[1]))
print("hysteresis     :", ctl.hysteresis.round(2))
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
# The NEXT model and the deadband fit were trained BEFORE the end-load /
# installation settings loaded (power cycle of 2026-08-20). A healthy zero on
# J2/J3 sits well under ~1 SDK N.m; far above that means the firmware state
# shifted the effort residuals and every fit trained on the old state is stale.
if np.abs(z[[1, 2]]).max() > 1.2:
    print("\n!! zero on J2/J3 is %.2f SDK N.m - far above the healthy band."
          % np.abs(z[[1, 2]]).max())
    print("!! The gravity fits predate the firmware end-load change; re-collect:")
    print("!!     python examples/next_collect.py && python examples/next_train.py")
    print("!!     python examples/next_fit_deadband.py")
    print("!! Continuing anyway - expect extra drift until then.\n")

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
