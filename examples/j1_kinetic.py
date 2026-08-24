"""Calibrate J1's KINETIC friction (same single direction as j1_stiction).

Protocol (J2-J6 pinned at the calibration pose, one TorqueSession):
  For each torque level u: park J1 at the range start, kick briefly above the
  calibrated max stiction to break away, then hold u constant while J1 sweeps.
  Record (t, q1); J1 is gravity-free, so   M*dv/dt = u - f0 - b*v.
  Fit in integral form over sliding windows (no differentiation of noise):
  M*(v2-v1) + f0*dt + b*(q2-q1) = u*dt  ->  M, f0 (Coulomb/kinetic), b
  (viscous); window residuals give f0(q1). Saves data/j1_kinetic.npz for
  examples/j1_slip_comp.py. Estimator validated on synthetic data.

THE ARM MOVES BY ITSELF (J1 sweeps repeatedly): CLEAR THE WORKSPACE.

    python examples/j1_kinetic.py
    python examples/j1_kinetic.py --levels 0.5 0.7 0.9
"""
import argparse
import os
import sys
import time

_PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
try:
    from piperx_teleop import MitCommand, PiperModel, TorqueSession, require_patched_sdk
    require_patched_sdk()
except (RuntimeError, ImportError):
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

import numpy as np

RAD = np.pi / 180.0
BASE = np.radians([0.0, 45.0, -70.0, 0.0, 10.0, 0.0])
PIN_KP, PIN_KD = 3.0, 0.8
KICK_S = 0.08
V_STOP = 1.2          # abort a trial past this, rad/s (VMAX watchdog is 3.0)
TRIAL_S = 15.0

ap = argparse.ArgumentParser()
ap.add_argument("--levels", type=float, nargs="+", default=[0.35, 0.40, 0.45, 0.55],
                help="constant torques, N.m - keep JUST above kinetic f0 (~0.32): "
                     "b is tiny, so anything much higher accelerates to the "
                     "velocity watchdog instead of settling")
ap.add_argument("--stiction", default="data/j1_stiction.npz")
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

st = np.load(a.stiction, allow_pickle=True)
ok = np.isfinite(st["tau_break"])
GRID, TAUS = st["q1"][ok], st["tau_break"][ok]
DIR = float(st["direction"])
KICK = 1.15 * TAUS.max()
Q_START = GRID.min() if DIR > 0 else GRID.max()
Q_END = GRID.max() if DIR > 0 else GRID.min()
END_GUARD = Q_END - 8.0 * RAD * DIR
mdl = PiperModel()


class KinLaw:
    def __init__(self):
        self.mode = "pin"
        self.q_ref = None
        self.u = 0.0
        self.t0 = None

    def __call__(self, s):
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        tau = mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        if self.mode == "const":
            if self.t0 is None:
                self.t0 = s.t
            kp[0] = kd[0] = 0.0
            tau[0] += (KICK if s.t - self.t0 < KICK_S else self.u) * DIR
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


law = KinLaw()
sess = TorqueSession(law, can=a.can)


def goto(tgt, secs=2.0):
    law.mode = "pin"
    law.t0 = None
    q0, t0 = (law.q_ref.copy() if law.q_ref is not None else sess.q()), time.time()
    while time.time() - t0 < secs + 0.4:
        f = min((time.time() - t0) / secs, 1.0)
        law.q_ref = q0 + f * (tgt - q0)
        time.sleep(0.01)


print("levels %s N.m, kick %.2f N.m for %.0f ms, sweep %.0f -> %.0f deg" % (
    a.levels, KICK, KICK_S * 1e3, np.degrees(Q_START), np.degrees(END_GUARD)))
if not a.yes:
    input(">>> CLEAR THE WORKSPACE - J1 sweeps %d times. ENTER to start " % len(a.levels))

trials = []
with sess:
    tgt = BASE.copy()
    tgt[0] = Q_START
    goto(tgt, secs=3.0)
    for u in a.levels:
        tgt = law.q_ref.copy()
        tgt[0] = Q_START
        goto(tgt, secs=max(1.0, abs(Q_START - sess.q()[0]) / (20 * RAD)))
        time.sleep(0.6)
        law.u = u
        law.mode = "const"
        rec_t, rec_q = [], []
        t0 = time.time()
        stall_q, stall_t = sess.q()[0], t0
        why = "end"
        while True:
            time.sleep(0.004)
            now = time.time()
            q1 = sess.q()[0]
            rec_t.append(now - t0)
            rec_q.append(q1)
            if (q1 - END_GUARD) * DIR > 0:
                break
            if abs(q1 - stall_q) > 0.5 * RAD:
                stall_q, stall_t = q1, now
            elif now - stall_t > 1.2 and now - t0 > KICK_S + 0.5:
                why = "stall"
                break
            if now - t0 > TRIAL_S:
                why = "timeout"
                break
            if len(rec_q) > 6:
                dt_w = rec_t[-1] - rec_t[-6]
                if dt_w > 1e-3 and abs(rec_q[-1] - rec_q[-6]) / dt_w > V_STOP:
                    why = "fast"
                    break
            if not sess.running:
                raise SystemExit("session tripped: %s" % sess.trip)
        law.mode = "pin"
        law.t0 = None
        law.q_ref = sess.q()
        trials.append((u, np.array(rec_t), np.array(rec_q)))
        print("  u %.2f N.m: %5.1f deg in %4.1f s (%s)" % (
            u, np.degrees(abs(rec_q[-1] - rec_q[0])), rec_t[-1], why))
    goto(np.concatenate([[0.0], BASE[1:]]), secs=3.0)

# ---- fit (integral form, no differentiation of noise):
#      M*(v2-v1) + f0*(t2-t1) + b*(q2-q1) = u*(t2-t1)  over sliding windows ----
X_, y_, qm_ = [], [], []
for u, t, q in trials:
    if len(t) < 80:
        continue

    def vloc(i):
        sl = slice(max(i - 6, 0), i + 7)
        return np.polyfit(t[sl], q[sl], 1)[0] * DIR

    K = max(10, int(0.2 / max(np.median(np.diff(t)), 1e-3)))   # ~200 ms windows
    i0 = np.searchsorted(t, KICK_S + 0.1)
    for i in range(i0, len(t) - K - 8, max(K // 4, 1)):
        v1, v2 = vloc(i), vloc(i + K)
        if min(v1, v2) < 0.05:
            continue
        dt = t[i + K] - t[i]
        X_.append([v2 - v1, dt, (q[i + K] - q[i]) * DIR])
        y_.append(u * dt)
        qm_.append(0.5 * (q[i] + q[i + K]))
if len(y_) < 30:
    raise SystemExit("too few usable windows (%d) - trials stalled? raise --levels" % len(y_))
X, y, Qs = np.array(X_), np.array(y_), np.array(qm_)
(M, f0, b), *_ = np.linalg.lstsq(X, y, rcond=None)
res = (y - X @ [M, 0.0, b]) / X[:, 1]                # local f0 per window
bins = np.arange(GRID.min(), GRID.max() + 5 * RAD, 5 * RAD)
idx = np.digitize(Qs, bins)
q_bins = 0.5 * (bins[:-1] + bins[1:])
f0_bins = np.array([res[idx == i + 1].mean() if (idx == i + 1).sum() > 3 else np.nan
                    for i in range(len(q_bins))])
V_ = [X[:, 2] / X[:, 1]]                             # mean v per window, for the report

print("\nJ1 kinetic (%s dir): f0 %.3f N.m  b %.3f N.m.s/rad  M %.3f kg.m2  "
      "(%d samples, v %.2f..%.2f rad/s)" % (
          "+" if DIR > 0 else "-", f0, b, M, len(y),
          V_[0].min(), V_[0].max()))
print("f0(q1):", " ".join("%.0fdeg:%.2f" % (np.degrees(qq), ff)
                          for qq, ff in zip(q_bins, f0_bins) if np.isfinite(ff)))
fs_interp = np.interp(q_bins, GRID, TAUS)
print("drop F_s - f0(q): mean %.2f N.m  -> this is what the slip transition must absorb"
      % np.nanmean(fs_interp - f0_bins))
np.savez("data/j1_kinetic.npz", M=M, f0=f0, b=b, q_bins=q_bins, f0_bins=f0_bins,
         direction=DIR, levels=a.levels,
         **{"trial_%d_%s" % (i, k): v for i, (u, t, q) in enumerate(trials)
            for k, v in [("u", u), ("t", t), ("q", q)]})
print("saved data/j1_kinetic.npz")
print("position hold restored" + (" (%s)" % sess.trip if sess.trip else ""))
