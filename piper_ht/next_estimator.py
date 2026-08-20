"""Run-time NEXT estimator: predicts contact-free effort from recent history.

Stateless sliding window, as in FACTR 2 - the LSTM sees the last H samples of
[q, qd, q_cmd - q] and its hidden state is reset every call, so there is no
drift and no warm-up beyond filling the buffer.

Until the buffer fills (H samples, 0.5 s at 100 Hz) it falls back to the
analytic gravity model, so a controller can start immediately.
"""

import collections
import os

import numpy as np

from .gravity_fit import MAX_PLAUSIBLE_EFFORT

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "next_model.pt")


class NextEstimator:
    def __init__(self, path=MODEL_PATH, fallback=None, device="cpu"):
        import torch
        import torch.nn as nn
        self.torch = torch
        ck = torch.load(path, map_location=device, weights_only=False)
        self.H = int(ck["hist"])
        self.mu, self.sd = ck["mu"], ck["sd"]
        self.ymu, self.ysd = ck["ymu"], ck["ysd"]
        self.rms = ck.get("rms")
        # Models trained with --no-cmd take only [q, qd].  Feeding q_cmd back in
        # closes a loop when the consumer is an admittance controller, since
        # q_cmd is then that controller's own output.
        self.use_cmd = bool(ck.get("use_cmd", True))
        self.din = int(ck.get("din", 18))

        class NEXT(nn.Module):
            def __init__(self, din=18, dout=6, hid=128, mlp=256, drop=0.1):
                super().__init__()
                self.lstm = nn.LSTM(din, hid, num_layers=2, batch_first=True, dropout=drop)
                self.head = nn.Sequential(nn.Linear(hid, mlp), nn.ReLU(), nn.Dropout(drop),
                                          nn.Linear(mlp, mlp), nn.ReLU(), nn.Dropout(drop),
                                          nn.Linear(mlp, dout))

            def forward(self, x):
                o, _ = self.lstm(x)
                return self.head(o[:, -1])

        self.net = NEXT(din=self.din).to(device)
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.device = device
        self.buf = collections.deque(maxlen=self.H)
        self.fallback = fallback

    def reset(self):
        self.buf.clear()

    def ready(self):
        return len(self.buf) == self.H

    def push(self, q, qd, q_cmd):
        """Append one sample and return the predicted contact-free effort."""
        if self.use_cmd:
            self.buf.append(np.concatenate([q, qd, np.asarray(q_cmd) - np.asarray(q)]))
        else:
            self.buf.append(np.concatenate([q, qd]))
        if not self.ready():
            return self.fallback.effort(q) if self.fallback is not None else np.zeros(6)
        x = (np.asarray(self.buf, dtype=np.float32) - self.mu) / self.sd
        with self.torch.no_grad():
            t = self.torch.from_numpy(x[None]).to(self.device)
            y = self.net(t).cpu().numpy()[0]
        out = y * self.ysd + self.ymu
        # Same guard as the analytic model: a prediction beyond anything the
        # arm can physically need is a model failure, not a real load.
        return np.clip(out, -MAX_PLAUSIBLE_EFFORT, MAX_PLAUSIBLE_EFFORT)
