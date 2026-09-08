"""Per-joint gravity correction (b601's g_scale / g_bias) from torque-mode residuals.

The breakaway brackets in friction_cal.py (static rows) and tool_id.py (torque step)
both measure, per joint and pose, (u+ + u-)/2 = the torque the model is missing, in
t_ff units. A rigid-body fit (tool_id) explains what varies with the wrist orientation;
this script fits what is left as

    tau_ff_j = scale_j * g_j(q) + bias_j

scale_j is kept only where it explains the pose-to-pose spread (>= 15 % rms reduction and
|corr| > 0.5), else 1. bias_j is the mean residual. Note a constant bias is also what
direction-asymmetric Coulomb friction looks like to these measurements, and for the
controller the two are the same thing, so friction_cal.py's static levels are symmetric
and the asymmetry lives here.

    python examples/gravity_cal.py                        # -> data/gravity_cal.npz
    python examples/gravity_cal.py --tool data/tool_body.npz --measured-with data/tool_prior.npz
    python examples/drag_mode.py --tool data/tool_body.npz --gravity data/gravity_cal.npz ...

--measured-with is the tool file the residual data were collected against (the residual
is re-expressed against --tool). Re-run this after any new friction_cal or tool_id run.
"""
import argparse
import csv
import os

import numpy as np

from piperx_teleop import model_with_tool

np.set_printoptions(precision=3, suppress=True, linewidth=150)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tool", default="data/tool_body.npz")
ap.add_argument("--measured-with", default="data/tool_prior.npz", help="tool file used during the runs")
ap.add_argument("--csv", default="data/friction_cal.csv")
ap.add_argument("--torque", default="data/tool_torque.npz")
ap.add_argument("--out", default="data/gravity_cal.npz")
a = ap.parse_args()

mdl = model_with_tool(a.tool if os.path.exists(a.tool) else None)
old = model_with_tool(a.measured_with if os.path.exists(a.measured_with) else None)
Q, R, MASK = [], [], []
if os.path.exists(a.csv):
    for r in csv.DictReader(open(a.csv, newline="")):
        if r["kind"] != "static":
            continue
        q = np.array([float(r["q%d" % (i + 1)]) for i in range(6)])
        res = np.array([float(r["resid%d" % (i + 1)]) for i in range(6)])
        Q.append(q); R.append(res); MASK.append(np.isfinite(res))
if os.path.exists(a.torque):
    t = np.load(a.torque)
    for q, j, res in zip(t["q"], t["joint"], t["residual"]):
        v = np.full(6, np.nan); v[j] = res
        Q.append(np.asarray(q, float)); R.append(v); MASK.append(np.isfinite(v))
Q, R, MASK = np.array(Q), np.array(R), np.array(MASK)
if len(Q) == 0:
    raise SystemExit("no residual data: run friction_cal.py run (static) and/or tool_id.py torque first")
# residuals were measured against the OLD tool model; re-express against the current one
R = R - np.array([mdl.gravity_torque_raw(q) - old.gravity_torque_raw(q) for q in Q])
G = np.array([mdl.gravity_torque_raw(q) for q in Q])

scale, bias = np.ones(6), np.zeros(6)
print("%d residual rows (tool %s)" % (len(Q), a.tool))
print("  joint  n   mean resid   rms before   scale   bias    rms after   corr(resid, g)")
for j in range(6):
    m = MASK[:, j]
    if m.sum() < 3:
        print("  J%d    %2d   (not enough data)" % (j + 1, m.sum())); continue
    g, r = G[m, j], R[m, j]
    rms0 = np.sqrt(np.mean(r ** 2))
    b = r.mean(); rms_b = np.std(r)
    s_j = 1.0
    if g.std() > 0.3:
        sl, c = np.polyfit(g, r, 1)
        corr = np.corrcoef(g, r)[0, 1]
        rms_s = np.std(r - (sl * g + c))
        if abs(corr) > 0.5 and rms_s < 0.85 * rms_b:
            s_j, b, rms_b = 1.0 + sl, c, rms_s
    else:
        corr = 0.0
    scale[j], bias[j] = s_j, b
    print("  J%d    %2d   %+7.3f      %6.3f      %.3f  %+7.3f    %6.3f      %+.2f" % (j + 1, m.sum(), r.mean(), rms0, s_j, b, rms_b, corr))
bias = np.clip(bias, -1.0, 1.0)
np.savez(a.out, scale=scale, bias=bias, tool=a.tool, n=len(Q))
print("\nsaved %s   scale %s   bias %s" % (a.out, scale.round(3), bias.round(3)))
print("use:  --tool %s --gravity %s" % (a.tool, a.out))
