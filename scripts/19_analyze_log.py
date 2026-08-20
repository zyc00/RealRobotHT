"""Read data/teleop_log.npz and say what actually happened."""
import sys
import numpy as np
d = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/teleop_log.npz")
t, cl, ok = d["t"], d["clutch"], d["valid"]
cpos, disp, goal, cmd, act, q = d["cpos"], d["disp"], d["goal"], d["cmd"], d["actual"], d["q"]
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
print("ticks %d over %.1f s (%.0f Hz) | clutched %.0f%% | tracking valid %.0f%%"
      % (len(t), t[-1], len(t) / max(t[-1], 1e-9), 100 * cl.mean(), 100 * ok.mean()))
if not cl.any():
    sys.exit("\nCLUTCH WAS NEVER ENGAGED - the squeeze never passed threshold.")
m = cl
print("\nwhile clutched:")
print("  controller moved      : %s mm (range per axis)" % ((cpos[m].max(0) - cpos[m].min(0)) * 1000).round(0))
print("  commanded disp (robot): %s mm" % ((disp[m].max(0) - disp[m].min(0)) * 1000).round(0))
print("  goal range            : %s mm" % ((goal[m].max(0) - goal[m].min(0)) * 1000).round(0))
print("  cmd  range            : %s mm" % ((cmd[m].max(0) - cmd[m].min(0)) * 1000).round(0))
print("  ACTUAL arm range      : %s mm" % ((act[m].max(0) - act[m].min(0)) * 1000).round(0))
print("  joint range (deg)     : %s" % np.degrees(q[m].max(0) - q[m].min(0)).round(1))
lag = np.linalg.norm(cmd[m] - act[m], axis=1)
print("\n  cmd-vs-actual lag: mean %.0f mm  p95 %.0f mm  max %.0f mm"
      % (lag.mean() * 1000, np.percentile(lag, 95) * 1000, lag.max() * 1000))
gl = np.linalg.norm(goal[m] - cmd[m], axis=1)
print("  goal-vs-cmd gap  : mean %.0f mm  max %.0f mm" % (gl.mean() * 1000, gl.max() * 1000))
print("\n  clamps: step %d  box %d  floor %d  rot %d"
      % (d["cl_step"][-1], d["cl_box"][-1], d["cl_floor"][-1], d["cl_rot"][-1]))
print("\ndiagnosis:")
cm = (cpos[m].max(0) - cpos[m].min(0)).max()
if cm < 0.02:
    print("  * the CONTROLLER barely moved (%.0f mm) - tracking or clutch problem" % (cm * 1000))
elif (disp[m].max(0) - disp[m].min(0)).max() < 0.02:
    print("  * controller moved but the mapped displacement did not - MAPPING bug")
elif (goal[m].max(0) - goal[m].min(0)).max() < 0.02:
    print("  * displacement present but goal did not move - gain/anchor bug")
elif (cmd[m].max(0) - cmd[m].min(0)).max() < 0.02:
    print("  * goal moved but command did not - rate limit or leash pinned it")
elif (act[m].max(0) - act[m].min(0)).max() < 0.02:
    print("  * command moved but the ARM did not - firmware refusing/stalled")
else:
    print("  * every stage moved; ratio arm/controller = %.2f"
          % ((act[m].max(0) - act[m].min(0)).max() / cm))
