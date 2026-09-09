"""Analyse a drag_mode.py --log file: per-joint oscillation frequency and where it enters.

    python examples/log_analyze.py data/drag_log.npz [more.npz ...]

For each joint: dominant frequency of velocity, of the friction feed-forward, of the
shaping torque and of the damping-inclusive total, the fraction of power at that
frequency, and the correlation of each torque term with velocity (positive = pushing
along the motion, i.e. the term is feeding the oscillation). Also reports the observer
residual statistics at rest and the sign of feedback between shaping torque and velocity.
"""
import sys

import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=170)


def peak(x, dt):
    x = x - x.mean()
    if x.std() < 1e-9:
        return 0.0, 0.0
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), dt)
    m = fr > 0.3
    i = np.argmax(X[m])
    return fr[m][i], X[m][i] / max(X[m].sum(), 1e-12)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    s = a.std() * b.std()
    return float((a * b).mean() / s) if s > 1e-12 else 0.0


for path in sys.argv[1:]:
    d = np.load(path)
    A = d["data"]
    if 'controller' in d and str(d['controller']) == 'b601-reference':
        if len(A) < 3:
            print(path, ': insufficient reference samples'); continue
        dt = float(np.median(np.diff(A[:, 0])))
        print(path, ': B601 reference; %.1f s, median tick %.2f ms, alpha %.2f..%.2f' %
              (A[-1, 0]-A[0, 0], dt*1000, A[:, 43].min(), A[:, 43].max()))
        print(' joint | velocity std | peak Hz | residual std | combined assist std')
        for j in range(6):
            fq, _ = peak(A[:, 7+j], dt)
            print(' J%d | %.3f | %.2f | %.3f | %.3f' %
                  (j+1, A[:, 7+j].std(), fq, A[:, 25+j].std(), A[:, 31+j].std()))
        print('Combined assist includes friction, shaping, Cartesian damping and optional B601 breakaway.')
        continue
    t = A[:, 0]; q = A[:, 1:7]; qd_raw = A[:, 7:13]; qd = A[:, 13:19]; g = A[:, 19:25]
    f = A[:, 25:31]; tau = A[:, 31:37]; r = A[:, 37:43]; tb = A[:, 43:49]; alpha = A[:, 49]; cond = A[:, 50]
    dt = float(np.median(np.diff(t)))
    assist = A[:, 51:57] if A.shape[1] >= 57 else np.zeros_like(f)
    # tau_bal already includes Cartesian damping; this residual is linear kd.
    damp = tau - g - f - tb - assist
    print("=" * 100)
    print("%s: %d ticks, %.1f s, tick %.2f ms (p99 %.2f ms), alpha %.2f..%.2f, cond %.0f..%.0f" % (
        path, len(A), t[-1] - t[0], dt * 1e3, np.percentile(np.diff(t), 99) * 1e3, alpha.min(), alpha.max(), cond.min(), cond.max()))
    still = np.all(np.abs(qd) < 0.02, axis=1)
    if A.shape[1] >= 57:
        print("breakaway peak |torque| (N.m):", np.max(np.abs(assist), axis=0))
    if still.sum() > 50:
        print("at rest (%d ticks): r mean %s  r std %s" % (still.sum(), r[still].mean(0), r[still].std(0)))
    print("raw-velocity ticks that read exactly 0 (stale frame): %s %%" % (100 * np.mean(qd_raw == 0, axis=0)).round(0))
    print("\n joint | qd std  f(qd)Hz pow | f_ff std corr(v) | tau_bal std corr(v) f(Hz) | damp std corr(v) | r std")
    for j in range(6):
        fq, pq = peak(qd[:, j], dt); fb, pb = peak(tb[:, j], dt)
        print("  J%d   | %.3f  %5.2f  %3.0f%% | %.3f  %+.2f    | %.3f   %+.2f   %5.2f  | %.3f  %+.2f   | %.3f" % (
            j + 1, qd[:, j].std(), fq, 100 * pq, f[:, j].std(), corr(f[:, j], qd[:, j]),
            tb[:, j].std(), corr(tb[:, j], qd[:, j]), fb, damp[:, j].std(), corr(damp[:, j], qd[:, j]), r[:, j].std()))
    # worst 2 s window on J2 and the lag structure there
    j = 1; w = int(2.0 / dt)
    if len(A) > w + 10:
        e = [qd[i:i + w, j].std() for i in range(0, len(A) - w, max(w // 4, 1))]
        i0 = int(np.argmax(e)) * max(w // 4, 1); seg = slice(i0, i0 + w)
        fq, pq = peak(qd[seg, j], dt)
        print("\nworst 2 s window for J2 at t=%.1f s: velocity %.2f Hz (%.0f%% power), |qd| mean %.3f, alpha %.2f" % (
            t[i0] - t[0], fq, 100 * pq, np.abs(qd[seg, j]).mean(), alpha[seg].mean()))
        x = qd[seg, j] - qd[seg, j].mean()
        for name, y in (("f_ff", f[seg, j]), ("assist", assist[seg, j]), ("tau_bal", tb[seg, j]), ("damp", damp[seg, j]), ("r", r[seg, j])):
            y = y - y.mean()
            if y.std() < 1e-9:
                print("   %-8s flat" % name); continue
            c = np.correlate(y, x, "full") / (x.std() * y.std() * len(x)); lags = np.arange(-len(x) + 1, len(x))
            m = (lags > -60) & (lags < 60); k = np.argmax(np.abs(c[m]))
            print("   %-8s std %.3f  corr with qd %+.2f at lag %+d ticks (%+.0f ms; + = term lags velocity)" % (
                name, y.std(), c[m][k], lags[m][k], lags[m][k] * dt * 1e3))
