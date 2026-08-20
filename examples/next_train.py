"""Train a NEXT-style free-space torque estimator (FACTR 2, arXiv 2606.12406).

Step 2 of 3 - run examples/next_collect.py first.

Measured on this arm: the learned model beat the analytic one on every joint
(J2 0.37 -> 0.30, J4 0.28 -> 0.14 reported N.m), but most of the practical gain
came from noticing the error scales with ARM SPEED, not each joint's own speed -
hence the speed-dependent deadband in next_fit_deadband.py.

NOTE the --no-cmd flag. The paper feeds q_cmd - q as an input, which is right
when q_cmd comes from a human teleoperator. Fed to an admittance loop it closes
a feedback path - q_cmd is then that loop's own output - and the arm drifts.
Accuracy without it was unchanged, so --no-cmd is the safe default here.

Predicts the effort a contact-free arm should be drawing, from a short history
of proprioception.  External torque at run time is then the residual

    tau_ext = effort_measured - effort_predicted

which is what the admittance loop needs.  This replaces the analytic gravity
model, whose error (0.26-0.47 N.m) is 6-47x the arm's own repeatability and is
what forces the heavy deadband.

Inputs per timestep, following the paper: [q, qd, q_cmd - q]  (18 dims)
History H=50.  2-layer LSTM(128) -> 2-layer MLP(256) -> 6 efforts.
"""
import argparse, sys, time
import numpy as np
sys.path.insert(0, ".")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/freemotion.npz")
ap.add_argument("--out", default="data/next_model.pt")
ap.add_argument("--hist", type=int, default=50)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--no-cmd", action="store_true",
                help="drop the q_cmd-q input. Required when the consumer is an "
                     "admittance loop, because q_cmd is then the controller's own "
                     "output and feeding it back closes a loop through the model.")
a = ap.parse_args()

import torch
import torch.nn as nn

d = np.load(a.data)
q, qd, q_cmd, eff = d["q"], d["qd"], d["q_cmd"], d["effort"]
N = len(q)
print("loaded %d samples (%.1f min at %.0f Hz)" % (N, N / float(d["rate"]) / 60, d["rate"]))

if a.no_cmd:
    X = np.concatenate([q, qd], axis=1).astype(np.float32)          # (N,12)
else:
    X = np.concatenate([q, qd, q_cmd - q], axis=1).astype(np.float32)  # (N,18)
DIN = X.shape[1]
Y = eff.astype(np.float32)                                          # (N,6)
H = a.hist

# Window starts, then split by contiguous blocks so no window straddles the
# train/val boundary and validation is not just a shifted copy of training.
starts = np.arange(0, N - H)
block = int(float(d["rate"]) * 10)          # ~10 s blocks
bid = starts // block
ub = np.unique(bid)
rng = np.random.default_rng(0)
val_b = set(rng.choice(ub, size=max(1, int(0.2 * len(ub))), replace=False).tolist())
tr_idx = starts[~np.isin(bid, list(val_b))]
va_idx = starts[np.isin(bid, list(val_b))]
print("windows: %d train / %d val (%d blocks)" % (len(tr_idx), len(va_idx), len(ub)))

mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-6
ymu, ysd = Y[tr_idx].mean(0), Y[tr_idx].std(0) + 1e-6
Xn = ((X - mu) / sd).astype(np.float32)
Yn = ((Y - ymu) / ysd).astype(np.float32)

dev = "cuda" if torch.cuda.is_available() else "cpu"
Xt = torch.from_numpy(Xn).to(dev)
Yt = torch.from_numpy(Yn).to(dev)
print("device:", dev)


def batch_windows(idx):
    off = torch.arange(H, device=dev)
    rows = torch.from_numpy(idx).to(dev)[:, None] + off[None, :]
    return Xt[rows], Yt[rows[:, -1]]


class NEXT(nn.Module):
    def __init__(self, din=DIN, dout=6, hid=128, mlp=256, drop=0.1):
        super().__init__()
        self.lstm = nn.LSTM(din, hid, num_layers=2, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hid, mlp), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(mlp, mlp), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(mlp, dout))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])


model = NEXT().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
lossf = nn.MSELoss()

best, best_state, bad = np.inf, None, 0
t0 = time.time()
for ep in range(a.epochs):
    model.train()
    perm = rng.permutation(len(tr_idx))
    tot = 0.0
    for k in range(0, len(perm), a.batch):
        xb, yb = batch_windows(tr_idx[perm[k:k + a.batch]])
        opt.zero_grad()
        l = lossf(model(xb), yb)
        l.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += l.item() * len(xb)
    model.eval()
    with torch.no_grad():
        vs, vn = 0.0, 0
        for k in range(0, len(va_idx), 4096):
            xb, yb = batch_windows(va_idx[k:k + 4096])
            vs += lossf(model(xb), yb).item() * len(xb); vn += len(xb)
        v = vs / vn
    sched.step(v)
    if v < best - 1e-5:
        best, bad = v, 0
        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    else:
        bad += 1
    if ep % 5 == 0 or bad == 0:
        print("  epoch %3d  train %.5f  val %.5f%s" % (ep, tot / len(perm), v, "  *" if bad == 0 else ""))
    if bad >= a.patience:
        print("  early stop at epoch %d" % ep); break
print("trained in %.1f s" % (time.time() - t0))

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    preds = []
    for k in range(0, len(va_idx), 4096):
        xb, _ = batch_windows(va_idx[k:k + 4096])
        preds.append(model(xb).cpu().numpy())
P = np.concatenate(preds) * ysd + ymu
Ytrue = Y[va_idx + H - 1]
rms = np.sqrt(((P - Ytrue) ** 2).mean(0))

# Baseline: the analytic gravity model on the same validation samples.
from piper_ht.gravity_fit import GravityFit
g = GravityFit()
sub = np.arange(0, len(va_idx), max(1, len(va_idx) // 3000))
B = np.array([g.effort(q[va_idx[i] + H - 1]) for i in sub])
brms = np.sqrt(((B - Y[va_idx[sub] + H - 1]) ** 2).mean(0))

rep = np.array([0.005, 0.041, 0.022, 0.010, 0.018, 0.015])
print("\nvalidation RMS, reported-effort N.m")
print("  joint   analytic model   NEXT      repeatability   NEXT/repeat")
for j in range(6):
    print("  J%d        %9.3f  %8.3f   %11.3f   %8.1fx"
          % (j + 1, brms[j], rms[j], rep[j], rms[j] / max(rep[j], 1e-6)))

torch.save({"state": best_state, "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd,
            "hist": H, "rms": rms, "din": DIN, "use_cmd": not a.no_cmd}, a.out)
print("\nsaved", a.out)
