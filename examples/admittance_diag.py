"""Why did the arm shake? Read back the last admittance run, per joint.

Two different failures feel like "shaking" and they need opposite fixes, so the
point of this script is to tell them apart:

  STUTTER - the joint keeps falling back through its deadband.  Torque builds
    until it clears the break-in threshold, the setpoint lurches, the position
    loop catches up, the residual collapses below the keep-moving threshold and
    it stalls until your push rebuilds it.  Position climbs smoothly; the
    COMMAND is what is chopped up.  Fix: LOWER `hysteresis` on that joint so it
    stays engaged through the dips, and add `mass` to smooth the lurch.

  CREEP - torque noise alone sits above the keep-moving threshold, so once the
    joint breaks away nothing ever shuts it off and it wanders with no hand on
    it.  Fix: RAISE `hysteresis` back above the noise floor.

Everything is recovered from the log itself, not from the config file, since the
config may have been retuned since the run:  qdot = (|tau| - thr)/D means a
regression of |tau| on |qdot| gives D as the slope and the keep-moving threshold
as the intercept.  It fits to R^2 = 1.00 because that is literally the law.

  python examples/admittance_diag.py [data/admittance_run.npz]
"""
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "data/admittance_run.npz"

d = np.load(RUN)
t, Q, U, V = d["t"], d["q"], d["tau_user"], d["qdot"]
n = len(t)
span = t[-1] - t[0]
fs = (n - 1) / span
print("%s: %d samples, %.1f s, %.0f Hz\n" % (RUN, n, span, fs))

rows = []
for j in range(6):
    v, u = V[:, j], U[:, j]
    on = np.abs(v) > 1e-4
    free = on & (np.abs(v) < 0.79)          # drop the v_max clamp, it is not linear
    if free.sum() < 50:
        rows.append(None)
        continue

    A = np.column_stack([np.abs(v[free]), np.ones(free.sum())])
    (D, thr), *_ = np.linalg.lstsq(A, np.abs(u[free]), rcond=None)

    # Break-in is whatever the torque had reached on the first tick of a burst.
    edges = np.diff(on.astype(int))
    starts = np.where(edges == 1)[0] + 1
    breakin = np.median(np.abs(u[starts])) if len(starts) else np.nan
    # How long it stays engaged before falling back through the deadband. This
    # is the stutter measure; a dropout RATE over the whole run is diluted by
    # however much of the session the joint sat idle.
    ends = np.where(edges == -1)[0] + 1
    if len(ends) and len(starts) and ends[0] < starts[0]:
        ends = ends[1:]
    m = min(len(starts), len(ends))
    burst = np.median(t[ends[:m]] - t[starts[:m]]) if m else np.nan

    # Split the command into intended drag (<1 Hz) and shake (>1 Hz).
    k = max(int(fs), 3)
    slow = np.convolve(v, np.ones(k) / k, mode="same")
    shake = (v - slow)[on].std() / max(slow[on].std(), 1e-9)

    # Sensor/model noise: whatever a short moving average cannot explain.
    noise = (u - np.convolve(u, np.ones(21) / 21, mode="same")).std()
    dropouts = len(starts) / span

    rows.append(dict(j=j, D=D, thr=thr, breakin=breakin, shake=shake,
                     noise=noise, drop=dropouts, active=on.mean(), burst=burst,
                     travel=np.degrees(Q[:, j].max() - Q[:, j].min()),
                     lurch=np.degrees(np.abs(v[on]).max()) if on.any() else 0.0))

print("  joint travel  D    break-in  keep-moving  noise   engaged  shake/  peak")
print("        (deg)              thr        thr     rms    burst(s)  drag   deg/s")
for r in rows:
    if r is None:
        continue
    print("   J%d %6.1f %5.2f %8.2f %11.2f %7.3f %8.1f %7.2f %6.0f"
          % (r["j"] + 1, r["travel"], r["D"], r["breakin"], r["thr"],
             r["noise"], r["burst"], r["shake"], r["lurch"]))

print()
verdict = False
for r in rows:
    if r is None or r["travel"] < 2.0:
        continue
    if r["noise"] > r["thr"]:
        verdict = True
        print("J%d CREEPS: noise %.3f exceeds the keep-moving threshold %.2f."
              % (r["j"] + 1, r["noise"], r["thr"]))
        print("   raise hysteresis on J%d to at least %.2f."
              % (r["j"] + 1, min(1.0, r["noise"] / max(r["breakin"], 1e-6) * 1.3)))
    elif r["shake"] > 1.5 and r["burst"] < 0.15 and r["travel"] > 5:
        verdict = True
        print("J%d STUTTERS: stays engaged only %.0f ms at a time, and %.1fx more of"
              % (r["j"] + 1, 1000 * r["burst"], r["shake"]))
        print("   the command is jitter", end="")
        print(" than motion. It banks %.2f N.m before breaking" % r["breakin"])
        print("   away, then lurches to", end="")
        print(" %.0f deg/s. Lower hysteresis on J%d toward %.2f"
              % (r["lurch"], r["j"] + 1, max(r["noise"] * 1.15 / r["breakin"], 0.1)))
        print("   (keeps it engaged, but", end="")
        print(" must stay above the %.3f noise floor), and add" % r["noise"])
        print("   mass %.2f for an 80 ms velocity filter." % (0.08 * r["D"]))
if not verdict:
    print("no joint is stuttering or creeping. If it still felt rough, it is the")
    print("drag itself, not the controller.")
