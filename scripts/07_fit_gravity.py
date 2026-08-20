"""Fit the per-joint gravity model, anchored to the URDF.

Earlier this fit regularised toward ZERO, which is wrong: any parameter
direction the data does not excite drifts to whatever value minimises the
penalty, and those values can be enormous as long as they cancel on the
training set.  The identification held J1=J6=0 throughout, so the J6 direction
was completely unconstrained - and the resulting model predicted -2727 N.m on J5
at J6=-15 deg, which drove J5 into a runaway.

The fix: parameterise the fit as

    prediction = s * urdf_gravity(q)  +  Y(q) @ d

and penalise only `d`.  Where the data says nothing, d -> 0 and the model falls
back to the scaled URDF, which is always physically bounded.
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from piper_ht.model import PiperModel

LAM = 3e-2          # penalty on the deviation-from-URDF block

d = np.load("data/ident.npz")
Q, E = d["q"], d["effort"]
mdl = PiperModel()
links = mdl.identifiable_links()
N = len(Q)
Y = np.array([mdl.gravity_regressor(q, links) for q in Q])
U = np.array([mdl.gravity_torque(q) for q in Q])
beta = mdl.beta_urdf(links)

rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(N), 5)


def solve(Yb, u, y, lam):
    """[s, d, c] with the ridge applied to d only.

    A small floor is added on every diagonal for conditioning, and the scale
    column is pinned when the joint has no gravity signature at all (J1 and J6
    have vertical axes, so u is identically zero and s means nothing).
    """
    A = np.hstack([u[:, None], Yb, np.ones((len(y), 1))])
    P = np.full(A.shape[1], 1e-6 * len(y))
    P[1:1 + Yb.shape[1]] = lam * len(y)
    if u.std() < 1e-9:
        P[0] = 1e6 * len(y)          # no signature -> force s to 0
    return np.linalg.solve(A.T @ A + np.diag(P), A.T @ y)


scales, devs, consts, CVS = [], [], [], []
print("poses %d   ridge on deviation block: %.0e\n" % (N, LAM))
print("  joint   urdf-only   fitted   cross-val")
for j in range(6):
    idx = mdl.subtree_link_index(j, links)
    Yb, u, y = Y[:, j, idx], U[:, j], E[:, j]
    w = solve(Yb, u, y, LAM)
    pred = np.hstack([u[:, None], Yb, np.ones((N, 1))]) @ w
    if u.std() < 1e-9:                       # no gravity signature (J1, J6)
        rms_urdf = float(np.std(y))
    else:
        s_only = np.polyfit(u, y, 1)
        rms_urdf = np.sqrt(np.mean((y - np.polyval(s_only, u)) ** 2))
    rms_fit = np.sqrt(np.mean((y - pred) ** 2))
    err = []
    for f in folds:
        tr = np.setdiff1d(np.arange(N), f)
        wf = solve(Yb[tr], u[tr], y[tr], LAM)
        err.append(y[f] - np.hstack([u[f][:, None], Yb[f], np.ones((len(f), 1))]) @ wf)
    cv = np.sqrt(np.mean(np.concatenate(err) ** 2))
    print("  J%d    %9.3f %8.3f %11.3f" % (j + 1, rms_urdf, rms_fit, cv))
    CVS.append(cv)
    full = np.zeros(Y.shape[2]); full[idx] = w[1:-1]
    scales.append(w[0]); devs.append(full); consts.append(w[-1])

np.savez("data/grav_fit.npz", scale=np.array(scales), dev=np.array(devs),
         const=np.array(consts), links=np.array(links), beta_urdf=beta)
print("\nsaved data/grav_fit.npz")

# Repeat visits: same pose, reached from an unrelated direction.  The spread
# between them is the true noise floor - no model can do better, and 3x it is
# the smallest honest deadband.
pairs = []
for i in range(N):
    for k in range(i + 1, N):
        if np.abs(Q[i] - Q[k]).max() < np.radians(1.0):
            pairs.append((i, k))
if pairs:
    dif = np.array([E[i] - E[k] for i, k in pairs])
    rep = np.abs(dif).mean(0) / 2
    print("\nrepeatability from %d revisited pose pairs (reported N.m):" % len(pairs))
    print("   half-difference :", rep.round(3))
    print("\n  joint   model cv   repeatability   headroom   floor deadband(3x)")
    for j in range(6):
        cvj = [r for r in []]
        print("  J%d" % (j + 1), end="")
        print("     %7.3f      %9.3f      %6.2fx     %7.3f" % (
            CVS[j], rep[j], CVS[j] / max(rep[j], 1e-6), 3 * max(rep[j], 1e-6)))
else:
    print("\n(no repeated poses found in this dataset)")
