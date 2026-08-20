"""Fit the speed-dependent deadband for a trained NEXT model.

Model error grows with how fast the whole arm is moving (not with each joint's
own speed - other joints' motion couples in), so the deadband follows

    deadband_j(v) = K * (a_j + b_j * v),   v = max|qdot| over all joints

which keeps slow dragging light without letting fast motion cause false pushes.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, ".")

model_path = sys.argv[1] if len(sys.argv) > 1 else "data/next_model.pt"
out_path = sys.argv[2] if len(sys.argv) > 2 else "data/next_deadband.npz"
K = 3.0

d = np.load("data/freemotion.npz")
q, qd, q_cmd, eff = d["q"], d["qd"], d["q_cmd"], d["effort"]
N = len(q)
ck = torch.load(model_path, map_location="cpu", weights_only=False)
H = int(ck["hist"]); din = int(ck.get("din", 18)); use_cmd = bool(ck.get("use_cmd", True))


class NEXT(nn.Module):
    def __init__(self, din=din, dout=6, hid=128, mlp=256, drop=0.1):
        super().__init__()
        self.lstm = nn.LSTM(din, hid, num_layers=2, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hid, mlp), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(mlp, mlp), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(mlp, dout))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])


net = NEXT(); net.load_state_dict(ck["state"]); net.eval()
X = np.concatenate([q, qd] + ([q_cmd - q] if use_cmd else []), axis=1).astype(np.float32)
Xn = ((X - ck["mu"]) / ck["sd"]).astype(np.float32)

starts = np.arange(0, N - H)
rng = np.random.default_rng(0)
block = 1000
bid = starts // block
ub = np.unique(bid)
val_b = set(rng.choice(ub, size=int(0.2 * len(ub)), replace=False).tolist())
va = starts[np.isin(bid, list(val_b))]

off = np.arange(H)
P = []
with torch.no_grad():
    for k in range(0, len(va), 4096):
        P.append(net(torch.from_numpy(Xn[va[k:k + 4096][:, None] + off[None, :]])).numpy())
P = np.concatenate(P) * ck["ysd"] + ck["ymu"]
last = va + H - 1
err = P - eff[last]
v = np.abs(qd[last]).max(1)

A = np.zeros(6); B = np.zeros(6)
print("model: %s   (inputs %d, use_cmd=%s)" % (model_path, din, use_cmd))
print("rms_err(v) = a + b*v   (reported N.m)")
bins = np.linspace(0, 1.6, 11)
for j in range(6):
    x, c = [], []
    for i in range(len(bins) - 1):
        m = (v >= bins[i]) & (v < bins[i + 1])
        if m.sum() > 80:
            x.append(v[m].mean()); c.append(np.sqrt((err[m, j] ** 2).mean()))
    b, a = np.polyfit(np.array(x), np.array(c), 1)
    A[j], B[j] = max(a, 1e-3), max(b, 0.0)
    print("  J%d  a=%.3f b=%.3f  -> @rest %.3f  @0.3 %.3f" % (j + 1, A[j], B[j], A[j], A[j] + B[j] * 0.3))

np.savez(out_path, a=A, b=B, K=K)
print("\ndeadband = %.1f*(a+b*v)" % K)
print("  at rest      :", (K * A).round(2))
print("  dragging 0.3 :", (K * (A + B * 0.3)).round(2))
print("saved", out_path)
