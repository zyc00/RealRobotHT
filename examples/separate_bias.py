"""Separate breakaway and steady-motion residuals without changing robot calibration.

The legacy CSV has sweep centres, not per-sample poses or tool provenance.
Results are diagnostics, not an identified pure gravity correction.
"""
import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def vector(row, prefix):
    return np.array([float(row[prefix + str(j)]) for j in range(1, 7)])


def collect(rows):
    static = [(vector(r, 'q'), vector(r, 'resid'), vector(r, 'f'))
              for r in rows if r['kind'] == 'static']
    pending = defaultdict(lambda: {1: deque(), -1: deque()})
    pairs = []
    for row in rows:
        if row['kind'] != 'kin':
            continue
        speed = float(row['vset'])
        if speed == 0:
            continue
        key = (tuple(vector(row, 'q')), abs(speed))
        group = pending[key]
        group[1 if speed > 0 else -1].append(row)
        while group[1] and group[-1]:
            p, m = group[1].popleft(), group[-1].popleft()
            vp, vm = vector(p, 'mv'), vector(m, 'mv')
            fp, fm = vector(p, 'f'), vector(m, 'f')
            # Require both directions to actually move, with comparable speeds.
            ok = (np.isfinite(fp) & np.isfinite(fm) & np.isfinite(vp) & np.isfinite(vm)
                  & (vp > 0.001) & (vm < -0.001)
                  & (np.minimum(abs(vp), abs(vm)) >= 0.7 * np.maximum(abs(vp), abs(vm))))
            midpoint = np.where(ok, (fp + fm) / 2, np.nan)
            level = np.where(ok, (fp - fm) / 2, np.nan)
            pairs.append((np.array(key[0]), midpoint, level, (vp - vm) / 2, key[1]))
    unmatched = sum(len(g[1]) + len(g[-1]) for g in pending.values())
    return static, pairs, unmatched


def fit(q, residual, gravity):
    """Equal weight per pose; do not treat repeated speeds as independent poses."""
    result = dict(bias=np.full(6, np.nan), scale=np.full(6, np.nan),
                  scatter=np.full(6, np.nan), mean=np.full(6, np.nan),
                  n=np.zeros(6, int), poses=np.zeros(6, int))
    for j in range(6):
        good = np.isfinite(residual[:, j])
        result['n'][j] = good.sum()
        grouped = defaultdict(list)
        for i in np.flatnonzero(good):
            grouped[tuple(q[i])].append(i)
        result['poses'][j] = len(grouped)
        if not grouped:
            continue
        r = np.array([np.mean(residual[idx, j]) for idx in grouped.values()])
        g = np.array([np.mean(gravity[idx, j]) for idx in grouped.values()])
        result['mean'][j] = r.mean()
        if len(grouped) < 3:
            continue
        bias, slope = r.mean(), 0.0
        if len(grouped) >= 4 and g.std() > 0.3 and r.std() > 1e-9:
            sl, intercept = np.polyfit(g, r, 1)
            if abs(np.corrcoef(g, r)[0, 1]) > 0.5 and np.std(r - sl * g) < 0.85 * r.std():
                slope, bias = sl, intercept
        result['bias'][j], result['scale'][j] = bias, 1 + slope
        result['scatter'][j] = np.std(r - slope * g - bias)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', default='data/friction_cal.csv')
    ap.add_argument('--tool', default='data/tool_body.npz')
    ap.add_argument('--measured-with', required=True,
                    help='tool model used to collect CSV; legacy data does not record it')
    ap.add_argument('--out', default='data/bias_separation.npz')
    args = ap.parse_args()
    from piperx_teleop import model_with_tool
    for path in (args.csv, args.tool, args.measured_with):
        if not Path(path).is_file():
            ap.error('missing input: ' + path)
    output = Path(args.out)
    report_path = output.with_suffix('.md')
    if output.exists() or report_path.exists():
        ap.error('output exists; use a new --out path to preserve previous results')
    with open(args.csv, newline='') as file:
        static, moving, unmatched = collect(list(csv.DictReader(file)))
    if not static or not moving:
        ap.error('need both static and paired kinetic rows')
    current, old = model_with_tool(args.tool), model_with_tool(args.measured_with)
    saved = {}
    fits = {}
    for name, samples in [('static', static), ('motion', moving)]:
        q = np.array([s[0] for s in samples])
        g = np.array([current.gravity_torque_raw(x) for x in q])
        old_g = np.array([old.gravity_torque_raw(x) for x in q])
        residual = np.array([s[1] for s in samples]) - (g - old_g)
        fits[name] = fit(q, residual, g)
        saved.update({name + '_' + k: v for k, v in fits[name].items()})
        saved[name + '_q'] = q
        saved[name + '_residual'] = residual
        saved[name + '_half_difference'] = np.array([s[2] for s in samples])
    saved['motion_speed'] = np.array([s[3] for s in moving])
    saved['motion_requested_speed'] = np.array([s[4] for s in moving])
    # Compare regimes at MATCHING sweep centres, cancelling common tool shifts.
    delta = np.full(6, np.nan)
    matched = np.zeros(6, int)
    for j in range(6):
        means = []
        for name in ('static', 'motion'):
            groups = defaultdict(list)
            for q, r in zip(saved[name + '_q'], saved[name + '_residual'][:, j]):
                if np.isfinite(r):
                    groups[tuple(q)].append(r)
            means.append({q: np.mean(r) for q, r in groups.items()})
        common = means[0].keys() & means[1].keys()
        matched[j] = len(common)
        if common:
            delta[j] = np.mean([means[0][q] - means[1][q] for q in common])
    saved.update(matched_static_minus_motion=delta, matched_poses=matched)
    lines = ['# Static and motion bias diagnostics', '',
             'Residual model: `(scale - 1) * g_raw(q) + bias`. Units: N·m. '
             'Fits weight each pose equally. Scatter is pose residual standard deviation, not a confidence interval.', '',
             '| Joint | Static n / poses | Static scale | Static bias | Static scatter | Motion n / poses | Motion scale | Motion bias | Motion scatter | Matched static − motion | Matched poses |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for j in range(6):
        s, m = fits['static'], fits['motion']
        lines.append(f'| J{j+1} | {s["n"][j]} / {s["poses"][j]} | {s["scale"][j]:.3f} | {s["bias"][j]:+.3f} | {s["scatter"][j]:.3f} | '
                     f'{m["n"][j]} / {m["poses"][j]} | {m["scale"][j]:.3f} | {m["bias"][j]:+.3f} | {m["scatter"][j]:.3f} | {delta[j]:+.3f} | {matched[j]} |')
    lines += ['', f'Inputs: `{args.csv}`; target tool `{args.tool}`; assumed acquisition tool `{args.measured_with}`.',
              f'Unmatched direction rows: {unmatched}. NaN means insufficient valid data (fewer than three poses for a fit).', '',
              'These are separate measured torque midpoints, not separately identified physical gravity and friction. '
              'Motion bias can still include directional kinetic friction and dynamic error. '
              'Static measurements include light firmware damping; moving sweeps use host PD and move multiple joints. '
              'Legacy CSV stores sweep centres, not actual sample poses, so the moving tool-model correction is approximate. '
              'Acquisition tool provenance is assumed from the supplied argument, not verified from the CSV.', '',
              'The static fit uses only friction-calibration rows to keep the comparison consistent; '
              'it does not mix in the separate tool-torque experiment. '
              'No existing gravity/friction file or controller setting is changed. '
              'This diagnostic NPZ intentionally has no generic scale/bias keys and is not a --gravity file.']
    np.savez(output, **saved, csv=args.csv, tool=args.tool, measured_with=args.measured_with,
             diagnostic_only=True)
    report_path.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved {output} and {report_path}')


if __name__ == '__main__':
    main()
