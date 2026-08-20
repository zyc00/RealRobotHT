"""Teleoperate a Piper arm and record the trajectory.

    python examples/record_teleop.py --out data/ep01.npz                       # keyboard
    python examples/record_teleop.py --out data/ep01.npz --source quest --calibrate
    python examples/record_teleop.py --out data/ep01.npz --source scripted     # no human

Config is loaded HERE and passed into the teleop objects; the piperx_teleop
package never reads a config file of its own.

One row per control tick.  `action` is the pose the arm was actually commanded
(replayable); `intent` is what the operator asked for before rate limiting and
clamping.  `clutch` marks the ticks that are demonstration - the rest is the
operator repositioning, and is usually dropped.
"""
import argparse
import os
import sys
import time

import numpy as np
from piperx_teleop import CartesianTeleop, PiperArm, TeleopSession, load_config
from piperx_teleop.sources import KeyboardSource, TeleopSample

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/episode.npz")
ap.add_argument("--can", default="can0")
ap.add_argument("--source", choices=["keyboard", "quest", "scripted"], default=None,
                help="overrides input.source in the config file")
ap.add_argument("--config", default="config/teleop.toml")
ap.add_argument("--home", default="data/home_pose.npz")
ap.add_argument("--unlock-rotation", action="store_true")
ap.add_argument("--calibrate", action="store_true", help="quest: measure your forward first")
ap.add_argument("--heading", type=float, default=0.0)
ap.add_argument("--seconds", type=float, default=16.0, help="scripted: duration")
ap.add_argument("--radius", type=float, default=0.05, help="scripted: circle radius (m)")
ap.add_argument("--save-home", action="store_true",
                help="save the arm's CURRENT joint pose as the home pose, then exit")
a = ap.parse_args()

if a.save_home:
    from piperx_teleop.arm import JOINT_LIMITS
    _arm = PiperArm(a.can).connect(require_control=False)
    q = _arm.q()
    m = np.degrees(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q))
    os.makedirs(os.path.dirname(a.home) or ".", exist_ok=True)
    np.savez(a.home, q=q)
    print("saved %s" % a.home)
    print("  joints (deg) :", np.degrees(q).round(1))
    print("  limit margins:", m.round(1))
    if m[3] < 25:
        print("  !! J4 is only %.0f deg from its limit. Cartesian range collapses there -"
              "\n     measured: J4 at 89 deg gave 27 mm of vertical travel, J4 near 0 gave"
              "\n     the full range. Consider rolling J4 toward 0 before saving." % m[3])
    if m.min() < 10:
        print("  !! J%d is only %.0f deg from its limit." % (int(m.argmin()) + 1, m.min()))
    sys.exit(0)


class ScriptedSource:
    """Traces a circle in the x-z plane. Validates the loop with no human in it,
    which a human demonstration cannot do - it is not reproducible."""

    def __init__(self, seconds, radius):
        self.T, self.r, self.t0 = seconds, radius, None

    def start(self):
        self.t0 = time.time()
        return self

    def stop(self):
        pass

    def poll(self):
        t = time.time() - self.t0
        s = TeleopSample(connected=True, clutch=True)
        if t > self.T:
            s.quit = True
            return s
        ramp = min(1.0, t / 2.0, max(0.0, (self.T - t) / 2.0))   # ease in and out
        ang = 2 * np.pi * (t / self.T) * 2.0
        s.disp_pos = np.array([self.r * np.sin(ang), 0.0, self.r * (1 - np.cos(ang))]) * ramp
        s.gripper_closed = (t % 8.0) > 4.0
        return s


cfg = load_config(a.config if os.path.exists(a.config) else None)
if a.unlock_rotation:
    cfg.rotation.unlock = True
source = a.source or cfg.input.source
print("config: %s | source: %s%s"
      % (a.config if os.path.exists(a.config) else "(packaged defaults)", source,
         "" if a.source else " (from config)"))

banner = "Ctrl-C to stop"
if source == "keyboard":
    src = KeyboardSource(cfg)
    from piperx_teleop.sources.keyboard import HELP
    banner = HELP
elif source == "scripted":
    cfg.motion.gain = 1.0            # the scripted path is already in robot metres
    src = ScriptedSource(a.seconds, a.radius)
    banner = "scripted %.0f s circle, r=%.0f mm" % (a.seconds, a.radius * 1000)
else:
    from piperx_teleop.sources import QuestSource, calibrate_forward
    src = QuestSource(cfg, heading_deg=a.heading).start()
    if not src.wait_connected(15.0):
        sys.exit("Quest not connected - is the RoboVR client running?")
    if a.calibrate:
        input(">>> Squeeze the grip, push your hand STRAIGHT AWAY from your chest,\n"
              "    release while still out there, then press ENTER. ")
        src.set_heading(calibrate_forward(src))
        print("forward heading: %.1f deg" % src.heading_deg)
    banner = "squeeze the right grip to clutch; Ctrl-C to stop"

if os.path.exists(a.home):
    home = np.load(a.home)["q"]
else:
    home = None
    print("!! no home pose at %s - the session will start wherever the arm happens to be,\n"
          "   so `max_reach` is measured from there. Set one with --save-home." % a.home)
arm = PiperArm(a.can).connect()
sess = TeleopSession(arm, CartesianTeleop(arm, cfg), src, home_q=home)

COLS = ["t", "t_mono", "obs_t", "obs_age", "clutch", "action", "intent", "lag",
        "q", "dq", "effort", "ee_pos", "ee_rpy", "gripper_pos"]
rec = {k: [] for k in COLS}

print(banner)
print("recording to %s\n" % a.out)
last = 0.0
try:
    for st in sess.step():
        rec["t"].append(st.t); rec["t_mono"].append(st.t_mono)
        rec["obs_t"].append(st.obs_t); rec["obs_age"].append(st.obs_age)
        rec["clutch"].append(st.clutch)
        rec["action"].append(st.action()); rec["intent"].append(st.intent())
        rec["lag"].append(st.lag)
        for k, v in st.observation().items():
            rec[k].append(v)
        if time.time() - last > 0.2:
            last = time.time()
            sys.stdout.write("\r  %5d ticks | %5d clutched | pos %s | lag %3.0f mm  "
                             % (len(rec["t"]), int(np.sum(rec["clutch"])),
                                st.actual.round(3), st.lag * 1000))
            sys.stdout.flush()
except KeyboardInterrupt:
    pass

if not rec["t"]:
    sys.exit("\nnothing recorded")

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
out = {k: np.array(v) for k, v in rec.items()}
out["rate"] = float(len(out["t_mono"]) / max(out["t_mono"][-1] - out["t_mono"][0], 1e-6))
out["q_start"] = out["q"][0]
np.savez(a.out, **out)

n, c = len(out["t"]), int(out["clutch"].sum())
dt = np.diff(out["t_mono"]) * 1e3
print("\n\nwrote %s" % a.out)
print("  %d ticks, %.1f s, %.0f Hz (jitter sd %.2f ms)"
      % (n, out["t_mono"][-1] - out["t_mono"][0], out["rate"], dt.std()))
print("  %d clutched (%.0f%%) - the rest is repositioning" % (c, 100 * c / n))
print("  EE travelled %s mm" % ((out["ee_pos"].max(0) - out["ee_pos"].min(0)) * 1000).round(0))
print("  lag mean %.1f mm  max %.1f mm | observation age mean %.2f ms"
      % (out["lag"].mean() * 1000, out["lag"].max() * 1000, out["obs_age"].mean() * 1e3))
hi = int((out["lag"] > 0.02).sum())
if hi:
    print("  %d ticks with lag > 20 mm - the arm was not keeping up there" % hi)
print("  stopped: %s" % sess.stopped_reason)
