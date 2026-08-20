"""Record a trajectory driven by a scripted source (no human in the loop).

Exists to validate the record -> replay loop end to end on hardware: a human
demonstration is not reproducible, so it cannot tell us whether a difference
came from the pipeline or from the operator.  A known path can.

Implements the same TeleopSource protocol the keyboard and Quest sources do.
"""
import argparse, os, sys, time
import numpy as np

from piperx_teleop import CartesianTeleop, PiperArm, TeleopSession, load_config
from piperx_teleop.sources import TeleopSample

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/scripted_ep.npz")
ap.add_argument("--can", default="can0")
ap.add_argument("--config", default="config/teleop.toml")
ap.add_argument("--home", default="data/home_pose.npz")
ap.add_argument("--seconds", type=float, default=16.0)
ap.add_argument("--radius", type=float, default=0.05)
a = ap.parse_args()


class ScriptedSource:
    """Traces a circle in the x-z plane, then returns to the start."""

    def __init__(self, seconds, radius):
        self.T = seconds
        self.r = radius
        self.t0 = None

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
        # ease in and out so we never demand a step-change in velocity
        ramp = min(1.0, t / 2.0, max(0.0, (self.T - t) / 2.0))
        ang = 2 * np.pi * (t / self.T) * 2.0
        s.disp_pos = np.array([self.r * np.sin(ang), 0.0, self.r * (1 - np.cos(ang))]) * ramp
        s.gripper_closed = (t % 8.0) > 4.0
        return s


cfg = load_config(a.config if os.path.exists(a.config) else None)
cfg.motion.gain = 1.0                       # scripted path is already in robot metres
home = np.load(a.home)["q"] if os.path.exists(a.home) else None

arm = PiperArm(a.can).connect()
ctl = CartesianTeleop(arm, cfg)
src = ScriptedSource(a.seconds, a.radius)
sess = TeleopSession(arm, ctl, src, home_q=home)

COLS = ["t", "t_mono", "obs_t", "obs_age", "clutch", "action", "intent", "lag",
        "q", "dq", "effort", "ee_pos", "ee_rpy", "gripper_pos"]
rec = {k: [] for k in COLS}
print("recording a %.0f s scripted circle (r=%.0f mm)..." % (a.seconds, a.radius * 1000))
for st in sess.step():
    rec["t"].append(st.t); rec["t_mono"].append(st.t_mono)
    rec["obs_t"].append(st.obs_t); rec["obs_age"].append(st.obs_age)
    rec["clutch"].append(st.clutch)
    rec["action"].append(st.action()); rec["intent"].append(st.intent())
    rec["lag"].append(st.lag)
    for k, v in st.observation().items():
        rec[k].append(v)

out = {k: np.array(v) for k, v in rec.items()}
out["rate"] = float(len(out["t_mono"]) / max(out["t_mono"][-1] - out["t_mono"][0], 1e-6))
out["q_start"] = out["q"][0]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez(a.out, **out)
n = len(out["t"])
print("\nwrote %s: %d ticks, %.1f s, %.0f Hz" % (a.out, n, out["t"][-1] - out["t"][0], out["rate"]))
print("  EE span (mm)      :", ((out["ee_pos"].max(0) - out["ee_pos"].min(0)) * 1000).round(0))
print("  lag: mean %.1f mm  max %.1f mm" % (out["lag"].mean() * 1000, out["lag"].max() * 1000))
dt = np.diff(out["t_mono"]) * 1e3
print("  tick interval: mean %.2f ms  max %.2f ms (jitter sd %.2f ms)" % (dt.mean(), dt.max(), dt.std()))
print("  observation age: mean %.2f ms  p95 %.2f ms" % (out["obs_age"].mean() * 1e3,
                                                        np.percentile(out["obs_age"], 95) * 1e3))
print("  stopped: %s" % sess.stopped_reason)
