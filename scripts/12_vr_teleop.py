"""Quest 3 controller -> Piper Cartesian teleoperation.

Depends only on `robovr` for Quest input.  Run --dry-run first: it exercises the
whole pipeline and prints the targets it WOULD command, without moving the arm.

  python scripts/12_vr_teleop.py --dry-run
  python scripts/12_vr_teleop.py                 # live

Controls (right controller):
  squeeze  hold to clutch - the arm only moves while held
  trigger  toggle gripper
  A        re-latch the frame heading from the headset
  B        re-latch the motion reference without moving
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.config import load as _load_config, describe as _cfg_path
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.cartesian import CartesianController, MOVE_P, MOVE_L
from piper_ht.quest_teleop import QuestCartesianSource, QuestTeleopConfig

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true", help="never command the arm")
ap.add_argument("--port", type=int, default=7777)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--no-adb", action="store_true")
ap.add_argument("--no-keep-awake", action="store_true",
                help="do not disable the headset's proximity sleep")
ap.add_argument("--rate", type=float, default=100.0)
ap.add_argument("--gain", type=float, default=1.0, help="controller metres -> robot metres")
ap.add_argument("--max-step", type=float, default=0.004)
ap.add_argument("--max-reach", type=float, default=0.30)
ap.add_argument("--min-z", type=float, default=0.05)
ap.add_argument("--speed", type=int, default=20)
ap.add_argument("--movel", action="store_true", help="MOVE L instead of MOVE P")
ap.add_argument("--yaw-offset", type=float, default=0.0)
ap.add_argument("--fixed-frame", action="store_true",
                help="ignore the headset and use the raw Quest world frame")
ap.add_argument("--frame-mode", choices=["between_clutch", "continuous", "latched"],
                default="between_clutch",
                help="how the frame heading follows the headset (worn: between_clutch)")
ap.add_argument("--unlock-rotation", action="store_true")
ap.add_argument("--rot-gain", type=float, default=0.4)
ap.add_argument("--rot-deadband", type=float, default=6.0,
                help="ignore wrist rotation below this many degrees")
ap.add_argument("--max-rot-step", type=float, default=1.5,
                help="max orientation change per cycle, degrees")
ap.add_argument("--max-joint-step", type=float, default=2.0)
ap.add_argument("--log", default="data/teleop_log.npz",
                help="per-tick trace written on exit (set empty to disable)")
ap.add_argument("--lead-limit", type=float, default=0.04,
                help="max distance the command may lead the arm's actual pose")
ap.add_argument("--save-home", action="store_true",
                help="capture the arm's CURRENT pose as the home pose, then exit")
ap.add_argument("--home", action="store_true",
                help="move to a well-conditioned start pose first (J5 off zero, "
                     "away from the wrist singularity) before teleoperating")
_cfg = _load_config(keys=set(vars(ap.parse_known_args()[0]).keys()))
if _cfg:
    ap.set_defaults(**_cfg)
a = ap.parse_args()

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

if not a.no_keep_awake and not a.no_adb:
    # The proximity sensor sleeps the headset the moment it comes off your head,
    # which stops OpenXR tracking and leaves the app connected but pose-less.
    # Verified working on Quest 3; does not survive a headset reboot.
    try:
        import subprocess
        subprocess.run(["adb", "shell", "am", "broadcast", "-a",
                        "com.oculus.vrpowermanager.prox_close"],
                       capture_output=True, timeout=8, check=False)
        # A placed headset that gets moved trips the Guardian boundary, which
        # drops to passthrough and pauses the app.  Suppress that too.
        for prop in ("guardian_pause", "guardian_disable"):
            subprocess.run(["adb", "shell", "setprop", "debug.oculus." + prop, "1"],
                           capture_output=True, timeout=8, check=False)
        print("headset: proximity sleep disabled, guardian paused")
    except Exception as e:
        print("could not disable proximity sleep (%s); the headset may sleep "
              "when set down" % e)

cfg = QuestTeleopConfig(host=a.host, port=a.port, adb_reverse=not a.no_adb,
                        yaw_offset_deg=a.yaw_offset,
                        head_yaw_align=not a.fixed_frame,
                        lock_rotation=not a.unlock_rotation)
cfg.frame_mode = a.frame_mode
cfg.rot_disp_deadband_deg = a.rot_deadband

# A measured body-forward beats anything derived from the headset: the headset
# points wherever it is set down, and when worn it follows your gaze rather than
# your body.  scripts/21_calib_yaw.py measures it from one deliberate reach.
_yaw_file = os.path.join("data", "yaw_calib.npz")
if os.path.exists(_yaw_file) and not a.fixed_frame:
    _h = float(np.load(_yaw_file)["heading_deg"])
    cfg.head_yaw_align = False
    cfg.head_relative = False
    cfg.frame_mode = "latched"
    src_heading = _h
    print("using MEASURED body frame: heading %.1f deg (data/yaw_calib.npz)" % _h)
else:
    src_heading = None
src = QuestCartesianSource(cfg)
if src_heading is not None:
    src.frame.set_yaw_offset(src_heading)
print("connecting to Quest on %s:%d (adb_reverse=%s)..." % (a.host, a.port, not a.no_adb))
try:
    src.start()
except Exception as e:
    sys.exit("could not start the Quest server: %s\n"
             "Is adb installed and the headset connected? (adb devices)" % e)

arm = PiperArm().connect(0.5)
if arm.in_teach_mode():
    sys.exit(
        "\nARM IS IN TEACHING MODE (%s) - it will ignore every command.\n"
        "This is not a teleop problem: JointCtrl and EndPoseCtrl are both\n"
        "silently discarded while in this mode.\n\n"
        "It is entered with the drag/teach button on the arm, and cannot be\n"
        "exited over CAN (grag_teach_ctrl 0x00/0x02/0x06 and track 0x04 were\n"
        "all tested and ignored). To fix:\n"
        "  1. press the teach button on the arm again, or\n"
        "  2. power-cycle the arm\n"
        "then re-run this script." % arm.control_mode())

if not a.dry_run and not all(arm.is_enabled()):
    print("enabling arm...")
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)

# Measured by scripts/18_descend_test.py: descending holds tool orientation, so
# the firmware's IK straightens the elbow and eventually jams a joint.  From
# J3=-80 it ran out of J3 travel after 336 mm; J3=-110 reaches 412 mm before J2
# binds instead.  More elbow bend than that does not help (J3=-135 gave 384 mm).
# Default fallback, measured by scripts/18_descend_test.py: descending holds the
# tool orientation, so the firmware's IK straightens the elbow until a joint
# jams.  From J3=-80 it ran out of J3 after 336 mm; J3=-110 reaches 412 mm.
HOME_Q = np.radians([0.0, 55.0, -110.0, 0.0, 30.0, 0.0])
_home_file = os.path.join("data", "home_pose.npz")
if os.path.exists(_home_file):
    HOME_Q = np.load(_home_file)["q"]
    print("home pose loaded from %s: %s" % (_home_file, np.degrees(HOME_Q).round(1)))
if a.save_home:
    q_now = arm.q()
    os.makedirs("data", exist_ok=True)
    np.savez(_home_file, q=q_now)
    print("saved current pose as home: %s" % np.degrees(q_now).round(1))
    sys.exit(0)

if a.home and not a.dry_run:
    print("homing to a well-conditioned pose", np.degrees(HOME_Q))
    t0 = time.time()
    while time.time() - t0 < 20:
        arm.move_j(HOME_Q, speed_pct=15)
        if np.abs(arm.q() - HOME_Q).max() < np.radians(1.0):
            break
        time.sleep(0.01)
    time.sleep(0.8)
    print("homed:", np.degrees(arm.q()).round(1))

ctl = CartesianController(arm, max_step_m=a.max_step, max_reach_m=a.max_reach,
                          min_z_m=a.min_z, max_joint_step_deg=a.max_joint_step,
                          speed_pct=a.speed,
                          move_mode=MOVE_L if a.movel else MOVE_P,
                          position_gain=a.gain, dry_run=a.dry_run,
                          lead_limit_m=a.lead_limit,
                          lock_rotation=not a.unlock_rotation,
                          max_rot_step_deg=a.max_rot_step)
ctl.start()
print("firmware end pose at start (m): %s  rot(deg): %s" % (ctl.origin.round(4), ctl.rot.round(1)))
print("rotation: %s" % ("UNLOCKED (euler %s, max %.1f deg/cycle)" % ("xyz", a.max_rot_step)
                        if a.unlock_rotation else "locked (position only)"))
print("frame mode: %s (heading follows the headset)" % a.frame_mode)
print("mode: %s | speed %d%% | max step %.0f mm | box +/-%.2f m | z floor %.2f m"
      % ("MOVE_L" if a.movel else "MOVE_P", a.speed, a.max_step * 1000, a.max_reach, a.min_z))
if a.dry_run:
    print(">>> DRY RUN - the arm will NOT be commanded <<<")
print("\nwaiting for the headset... (squeeze the right grip to clutch)\n")

dt = 1.0 / a.rate
last_print = 0.0
was_clutched = False
n = 0
rates = []
track_hist = []
last_track_warn = 0.0
TR = {k: [] for k in ("t", "clutch", "valid", "cpos", "cquat", "disp", "disprot",
                      "anchor", "goal", "cmd", "actual", "q", "hdg",
                      "cl_step", "cl_box", "cl_floor", "cl_rot")}
t_start = time.time()
try:
    while True:
        loop0 = time.time()
        s = src.poll()

        if not s.connected:
            if time.time() - last_print > 2.0:
                print("  quest: not connected  %s" % src.stats())
                last_print = time.time()
            time.sleep(0.1)
            continue

        if s.clutch and not was_clutched:
            ctl.relatch()
            print("  clutch ENGAGED  (heading %.0f deg, held for this drag)" % s.heading_deg)
        if was_clutched and not s.clutch:
            print("  clutch released")
        was_clutched = s.clutch

        if s.reset_reference:
            ctl.relatch()

        if s.clutch and not s.reset_reference:
            if not ctl.check_joints():
                print("\n!! ABORT: %s" % ctl.aborted)
                ctl.hold_joints()
                break
            ctl.follow(s.disp_pos, s.disp_rotvec * a.rot_gain)
            while ctl.notes:
                print("  !! %s" % ctl.notes.pop(0))

        # Tracking-health guard.  Loss of optical tracking is SILENT: the pose
        # stays valid and updates at full rate while the IMU dead-reckons, so
        # the arm just under-responds.  Watch how far the controller actually
        # travels while clutched and say so if it looks dead-reckoned.
        if s.clutch:
            track_hist.append(np.asarray(
                getattr(src._server.latest(), "right_grip").position, float)
                if getattr(src._server.latest(), "right_grip", None) is not None
                else np.zeros(3))
            if len(track_hist) > 400:
                track_hist.pop(0)
            if len(track_hist) == 400 and time.time() - last_track_warn > 8.0:
                span = (np.array(track_hist).max(0) - np.array(track_hist).min(0)).max()
                if span < 0.05:
                    last_track_warn = time.time()
                    print("\n  !! CONTROLLER BARELY MOVING: %.0f mm over 4 s of clutched "
                          "motion.\n     If your hand is moving more than that, the headset "
                          "cannot SEE the\n     controller and is dead-reckoning from the IMU. "
                          "Point the headset at\n     your hands (or wear it) - Quest 3 "
                          "controllers are tracked optically.\n" % (span * 1000))
        else:
            track_hist.clear()

        # Full per-tick trace: without this every diagnosis is guesswork.
        if a.log:
            try:
                act, _ = ctl.read_pose()
                grip = getattr(src._server.latest(), "right_grip", None)
                TR["t"].append(time.time() - t_start)
                TR["clutch"].append(bool(s.clutch))
                TR["valid"].append(bool(s.tracking_valid))
                TR["cpos"].append(np.asarray(grip.position, float) if grip is not None else np.zeros(3))
                TR["cquat"].append(np.asarray(grip.quat_xyzw, float) if grip is not None else np.zeros(4))
                TR["disp"].append(s.disp_pos.copy())
                TR["disprot"].append(s.disp_rotvec.copy())
                TR["anchor"].append(ctl.anchor_pos.copy() if ctl.anchor_pos is not None else np.zeros(3))
                TR["goal"].append((ctl.anchor_pos + s.disp_pos * a.gain) if ctl.anchor_pos is not None else np.zeros(3))
                TR["cmd"].append(ctl.cmd_pos.copy() if ctl.cmd_pos is not None else np.zeros(3))
                TR["actual"].append(act.copy())
                TR["q"].append(arm.q().copy())
                TR["hdg"].append(float(s.heading_deg))
                TR["cl_step"].append(ctl.clamp_step); TR["cl_box"].append(ctl.clamp_box)
                TR["cl_floor"].append(ctl.clamp_floor); TR["cl_rot"].append(ctl.clamp_rot)
            except Exception:
                pass

        ctl.set_gripper(s.gripper_closed)

        n += 1
        rates.append(time.time())
        if n % 50 == 0 and s.clutch:
            hz = 0.0
            if len(rates) > 10:
                hz = (len(rates) - 1) / max(rates[-1] - rates[0], 1e-6)
            rates.clear()
            terr, actual = ctl.tracking_error()
            print("  target %s | actual %s | lag %4.0f mm | hdg %4.0f | %3.0f Hz | "
                  "clamps step=%d box=%d floor=%d"
                  % (ctl.target.round(3), actual.round(3), terr * 1000,
                     s.heading_deg, hz, ctl.clamp_step, ctl.clamp_box, ctl.clamp_floor))
            if terr > 0.05:
                print("     ^ arm is NOT following the target (lag %.0f mm) - the firmware "
                      "may be refusing this pose" % (terr * 1000))

        time.sleep(max(0.0, dt - (time.time() - loop0)))
except KeyboardInterrupt:
    print("\ninterrupted")
finally:
    if not a.dry_run:
        ctl.hold_joints()
    src.stop()
    if a.log and TR["t"]:
        np.savez(a.log, **{k: np.array(v) for k, v in TR.items()})
        print("wrote %s  (%d ticks, %.1f s)" % (a.log, len(TR["t"]), TR["t"][-1]))
    print("stopped; arm holding position")
