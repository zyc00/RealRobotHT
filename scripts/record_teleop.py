"""Teleoperate and record a trajectory, using the piperx_teleop package.

    python scripts/record_teleop.py --out data/ep01.npz              # keyboard
    python scripts/record_teleop.py --out data/ep01.npz --source quest --calibrate

Records one row per control tick.  `action` is the commanded EE pose that the
arm was actually given (replayable); `intent` is what the operator asked for
before rate limiting and clamping.  `clutch` marks the ticks that are actually
demonstration - the rest are the operator repositioning their hand.
"""
import argparse
import os
import sys
import time

import numpy as np

from piperx_teleop import CartesianTeleop, PiperArm, TeleopSession, load_config
from piperx_teleop.sources import KeyboardSource

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/episode.npz")
ap.add_argument("--can", default="can0")
ap.add_argument("--source", choices=["keyboard", "quest"], default="keyboard")
ap.add_argument("--config", default="config/teleop.toml")
ap.add_argument("--home", default="data/home_pose.npz")
ap.add_argument("--unlock-rotation", action="store_true")
ap.add_argument("--calibrate", action="store_true", help="quest: measure your forward first")
ap.add_argument("--heading", type=float, default=0.0)
a = ap.parse_args()

cfg = load_config(a.config if os.path.exists(a.config) else None)
if a.unlock_rotation:
    cfg.rotation.unlock = True

if a.source == "keyboard":
    src = KeyboardSource(cfg)
    from piperx_teleop.sources.keyboard import HELP
    banner = HELP
else:
    from piperx_teleop.sources import QuestSource, calibrate_forward
    src = QuestSource(cfg, heading_deg=a.heading).start()
    if not src.wait_connected(15.0):
        sys.exit("Quest not connected - is the RoboVR client running?")
    if a.calibrate:
        input(">>> Squeeze the grip, push your hand STRAIGHT AWAY from your chest,\n"
              "    release while still out there, then press ENTER. ")
        h = calibrate_forward(src)
        src.set_heading(h)
        print("forward heading measured: %.1f deg" % h)
    banner = "squeeze the right grip to clutch; Ctrl-C to stop"

home = np.load(a.home)["q"] if os.path.exists(a.home) else None
arm = PiperArm(a.can).connect()
ctl = CartesianTeleop(arm, cfg)
sess = TeleopSession(arm, ctl, src, home_q=home)

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
        rec["action"].append(st.action())
        rec["intent"].append(st.intent())
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
print("\n\nwrote %s" % a.out)
print("  %d ticks over %.1f s (%.0f Hz)" % (n, out["t"][-1] - out["t"][0], out["rate"]))
print("  %d clutched (%.0f%%) - the rest is repositioning, usually dropped" % (c, 100 * c / n))
print("  EE travelled %s mm" % ((out["ee_pos"].max(0) - out["ee_pos"].min(0)) * 1000).round(0))
print("  stopped: %s" % sess.stopped_reason)
hi = int((out["lag"] > 0.02).sum())
if hi:
    print("  %d ticks with lag > 20 mm - the arm was not keeping up there" % hi)
