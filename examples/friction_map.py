"""Map static + kinetic friction for ALL joints across their range.

For each joint j, at a grid of stations q_j (others pinned at BASE):
  STICTION, both directions: slow torque ramp until breakaway, + then -.
    Gravity cancels in the mean:  F_s(q) = (u_break+ + u_break-) / 2
    and the half-difference (u+ - u-)/2 is the gravity-model RESIDUAL at q
    (bonus gravity-fit check, free from the same data).
  KINETIC, both directions: adaptive terminal-velocity sweep - step torque
    up, keep plateaued sub-V_STOP levels, fit the line  u = f0 + b*v
    (Coulomb f0 + viscous b). Self-tunes to each joint's f0; no runaway.

Saves data/friction_map.npz (per-joint arrays keyed j0..j5) and prints an
annotated table. Re-run a single joint with --joints to refine.

SAFETY (enforced by a live FK guard before every commanded move; a station
that would violate is skipped, not clamped):
  * table:   no link below MIN_Z (default 0.08 m; z=0 is the table)
  * forward: no link past MAX_X  (default 0.45 m)
Ranges are FK-clipped per joint at startup and printed before you confirm.

THE ARM MOVES ITSELF ACROSS WIDE RANGES ON EVERY JOINT: CLEAR THE WORKSPACE,
STAY ON THE E-STOP.

    python examples/friction_map.py                     # all 6 joints, ~20-30 min
    python examples/friction_map.py --joints 1 4        # just J1, J4
    python examples/friction_map.py --no-kinetic        # stiction map only (faster)
    python examples/friction_map.py --step 10           # coarser grid
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

sys.setswitchinterval(0.0005)
RAD = np.pi / 180.0
BASE = np.radians([0.0, 45.0, -70.0, 0.0, 10.0, 0.0])
PIN_KP, PIN_KD = 3.0, 0.8
RATE = 0.15                 # N.m/s ramp
BREAK_DEG = 0.3
JUMP_S = 0.3
KICK_S = 0.08
V_STOP = 1.2                # rad/s trial abort
KIN_S = 6.0                 # max one kinetic level
KIN_TRAVEL = 70.0           # deg; stop a kinetic drift after this much travel
MIN_Z, MAX_X = 0.08, 0.45   # safety envelope, metres

# per-joint: (grid_lo, grid_hi deg, ramp cap N.m, kinetic drift torque N.m)
PLAN = {
    0: (-55, 55, 1.5, 0.38),
    1: (20, 95, 4.0, 1.05),     # J2 heavy; FK will trim the forward end
    2: (-100, -22, 2.0, 0.45),
    3: (-58, 58, 1.2, 0.22),
    4: (-58, 58, 1.0, 0.18),
    5: (-85, 85, 0.8, 0.14),
}

ap = argparse.ArgumentParser()
ap.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
ap.add_argument("--step", type=float, default=7.5, help="deg between stations")
ap.add_argument("--no-kinetic", action="store_true")
ap.add_argument("--min-z", type=float, default=MIN_Z)
ap.add_argument("--max-x", type=float, default=MAX_X)
ap.add_argument("--can", default="can0")
ap.add_argument("--out", default="data/friction_map.npz")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()
MIN_Z, MAX_X = a.min_z, a.max_x
mdl = PiperModel()
LINKS = [k for k in mdl._walk(np.zeros(6))[0].keys() if k not in ("world", "base_link")]


def safe(q):
    """True iff no link violates the table / forward envelope at pose q."""
    T, _ = mdl._walk(q)
    for k in LINKS:
        p = T[k][:3, 3]
        if p[2] < MIN_Z or p[0] > MAX_X:
            return False
    return True


def safe_stations(j, lo, hi, step):
    grid = np.arange(lo, hi + 1e-6, step)
    out = []
    for qd in grid:
        q = BASE.copy()
        q[j] = qd * RAD
        if safe(q):
            out.append(qd * RAD)
    return np.array(out)


class MapLaw:
    """pin all at q_ref; optionally free joint `tj` with gravity + extra u."""

    def __init__(self):
        self.q_ref = None
        self.tj = None
        self.u = 0.0            # applied extra torque on tj (signed)
        self.kick = 0.0         # transient kick added while >0, decremented by caller

    def __call__(self, s):
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        tau = mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        if self.tj is not None:
            kp[self.tj] = kd[self.tj] = 0.0
            tau[self.tj] += self.u + self.kick
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


law = MapLaw()
sess = TorqueSession(law, can=a.can, hz=200.0)


def goto(tgt, secs=2.0):
    law.tj = None
    q0 = law.q_ref.copy() if law.q_ref is not None else sess.q()
    t0 = time.time()
    while time.time() - t0 < secs + 0.4:
        f = min((time.time() - t0) / secs, 1.0)
        law.q_ref = q0 + f * (tgt - q0)
        time.sleep(0.01)


def ramp_breakaway(j, cap, direction):
    """Free joint j, ramp |u| until it moves BREAK_DEG; return (u_break, jump)."""
    q_pin = sess.q()[j]
    law.u = 0.0
    law.tj = j
    t0 = time.time()
    u_break, jump = np.nan, np.nan
    while True:
        time.sleep(0.005)
        law.u = min(RATE * (time.time() - t0), cap) * direction
        dq = (sess.q()[j] - q_pin) * direction
        if dq > BREAK_DEG * RAD:
            u_break = abs(law.u)
            te = time.time()
            while time.time() - te < JUMP_S:
                jump = np.nanmax([jump, (sess.q()[j] - q_pin) * direction])
                time.sleep(0.005)
            break
        if abs(law.u) >= cap or not sess.running:
            break
    law.tj = None
    law.u = 0.0
    law.q_ref = sess.q()
    return u_break, jump


def kinetic_trial(j, u_drift, direction, cap):
    """Ramp gently to breakaway, then hold u_drift; record (t, q). Stops on
    travel cap (no joint-limit runaway), stall, V_STOP, unsafe, or timeout."""
    law.tj = j
    q_pin = sess.q()[j]
    tb = time.time()
    while True:                                       # ramp to breakaway
        time.sleep(0.004)
        law.u = min(RATE * (time.time() - tb), cap) * direction
        if (sess.q()[j] - q_pin) * direction > 0.4 * RAD:
            break
        if abs(law.u) >= cap or not sess.running or not safe(sess.q()):
            law.tj = None; law.u = 0.0; law.q_ref = sess.q()
            return np.array([]), np.array([])
    q_break = sess.q()[j]
    law.u = u_drift * direction
    rec_t, rec_q = [], []
    t0 = time.time()
    stall_q, stall_t = sess.q()[j], t0
    while True:
        time.sleep(0.004)
        now = time.time()
        q = sess.q()
        rec_t.append(now - t0)
        rec_q.append(q[j])
        if not safe(q) or not sess.running:
            break
        if abs(q[j] - q_break) > KIN_TRAVEL * RAD:
            break                                     # travel cap -> no runaway
        if abs(q[j] - stall_q) > 0.5 * RAD:
            stall_q, stall_t = q[j], now
        elif now - stall_t > 1.0 and now - t0 > 0.5:
            break                                     # stalled
        if now - t0 > KIN_S:
            break
        if len(rec_q) > 6:
            dt = rec_t[-1] - rec_t[-6]
            if dt > 1e-3 and abs(rec_q[-1] - rec_q[-6]) / dt > V_STOP:
                break
    law.tj = None; law.u = 0.0; law.q_ref = sess.q()
    return np.array(rec_t), np.array(rec_q)


def kinetic_sweep(j, direction, cap, f0_guess, stations):
    """Adaptive multi-level drift for an integral fit. Start below f0, step up,
    keep trials that travel 12..85 deg (enough data, no runaway). No plateau
    needed - the integral fit uses the acceleration transient."""
    trials = []
    u = 0.7 * f0_guess
    fine = max(0.02, 0.08 * f0_guess)
    tries, runaway, found = 0, 0, False
    while u < cap and len(trials) < 6 and tries < 22:
        tries += 1
        st0 = stations[0] if direction > 0 else stations[-1]
        tgt = law.q_ref.copy(); tgt[j] = st0
        goto(tgt, secs=max(1.0, abs(st0 - sess.q()[j]) / (15 * RAD)))
        time.sleep(0.5)                                # settle before breakaway
        t, q = kinetic_trial(j, u, direction, cap)
        mv = np.degrees(abs(q[-1] - q[0])) if len(q) else 0.0
        if mv < 3:
            tag = "stall"                              # below effective f0
            u += fine if found else max(fine, 0.13 * u)   # geometric escape
        elif mv < 12:
            if trials:
                tag = "fast"; runaway += 1; u += fine * 0.5
            else:
                tag = "crawl"; found = True; u += fine
        else:
            tag = "keep"; found = True; trials.append((u, t, q)); u += fine
        print("    u %.3f -> %5.1f deg  %s" % (u, mv, tag))
        if runaway >= 2:
            break
    if not trials:
        print("    (no sustained motion up to %.2f N.m - gravity-dominated?)" % u)
    return trials


def fit_kinetic(trials, direction):
    """Integral fit  M dv + f0 dt + b dq = u dt  over all levels -> f0, b, M."""
    X_, y_ = [], []
    for u, t, q in trials:
        if len(t) < 25:
            continue

        def vloc(i):
            sl = slice(max(i - 6, 0), i + 7)
            return np.polyfit(t[sl], q[sl], 1)[0] * direction

        K = max(8, int(0.15 / max(np.median(np.diff(t)), 1e-3)))
        for i in range(2, len(t) - K - 8, max(K // 4, 1)):
            v1, v2 = vloc(i), vloc(i + K)
            if min(v1, v2) < 0.03:
                continue
            dt = t[i + K] - t[i]
            X_.append([v2 - v1, dt, (q[i + K] - q[i]) * direction])
            y_.append(u * dt)
    if len(y_) < 12:
        return np.nan, np.nan, np.nan, len(y_)
    X, y = np.array(X_), np.array(y_)
    (M, f0, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    return f0, b, M, len(y)


JOINTS = [j - 1 for j in a.joints]
plan_ranges = {}
for j in JOINTS:
    lo, hi, cap, drift = PLAN[j]
    st = safe_stations(j, lo, hi, a.step)
    plan_ranges[j] = (st, cap, drift)
    if len(st):
        print("J%d: %2d stations %+.0f..%+.0f deg  (cap %.1f N.m%s)" % (
            j + 1, len(st), np.degrees(st[0]), np.degrees(st[-1]), cap,
            "" if len(st) == len(np.arange(lo, hi + 1e-6, a.step)) else ", FK-trimmed"))
    else:
        print("J%d: NO safe stations in range - skipped" % (j + 1))

if not a.yes:
    input(">>> CLEAR WORKSPACE, hand on E-STOP. ENTER to map %d joint(s) " % len(JOINTS))

result = {}
with sess:
    goto(BASE, secs=3.0)
    for j in JOINTS:
        stations, cap, drift = plan_ranges[j]
        if not len(stations):
            continue
        print("\n=== J%d ===" % (j + 1))
        q_grid, fs, resid, jmp_p, jmp_m = [], [], [], [], []
        kin_p, kin_m = [], []
        for st in stations:
            tgt = law.q_ref.copy()
            tgt[j] = st
            if not safe(tgt):
                continue
            goto(tgt, secs=max(0.8, abs(st - sess.q()[j]) / (15 * RAD)))
            time.sleep(0.5)
            up, jp = ramp_breakaway(j, cap, +1.0)
            goto(tgt, secs=1.2)                          # recentre
            time.sleep(0.4)
            um, jm = ramp_breakaway(j, cap, -1.0)
            goto(tgt, secs=1.2)
            q_grid.append(st)
            fs.append(np.nanmean([up, um]))
            resid.append((up - um) / 2 if np.isfinite(up) and np.isfinite(um) else np.nan)
            jmp_p.append(jp)
            jmp_m.append(jm)
            print("  q %+6.1f deg  F_s %s N.m   grav-resid %+5.2f   slip +%s/-%s deg" % (
                np.degrees(st),
                "%4.2f" % fs[-1] if np.isfinite(fs[-1]) else " >cap",
                resid[-1] if np.isfinite(resid[-1]) else np.nan,
                "%.1f" % np.degrees(jp) if np.isfinite(jp) else "-",
                "%.1f" % np.degrees(jm) if np.isfinite(jm) else "-"))

        if not a.no_kinetic:
            f0_guess = 0.35 * np.nanmedian(fs) if np.isfinite(np.nanmedian(fs)) else 0.3 * cap
            print("  kinetic + (f0 guess %.2f):" % f0_guess)
            kin_p = kinetic_sweep(j, +1.0, cap, f0_guess, stations)
            print("  kinetic - :")
            kin_m = kinetic_sweep(j, -1.0, cap, f0_guess, stations)
            f0p, bp, Mp, np_ = fit_kinetic(kin_p, +1.0)
            f0m, bm, Mm, nm_ = fit_kinetic(kin_m, -1.0)
            f0_true = np.nanmean([f0p, f0m])           # gravity cancels in the mean
            g_leak = (f0m - f0p) / 2 if np.isfinite(f0p) and np.isfinite(f0m) else np.nan
            print("  KINETIC + : f0 %.2f b %.3f (n%d)   - : f0 %.2f b %.3f (n%d)" % (
                f0p, bp, np_, f0m, bm, nm_))
            print("  -> gravity-cancelled kinetic f0 %.2f N.m   (gravity leak %.2f)" % (
                f0_true, g_leak))
        else:
            f0p = bp = Mp = f0m = bm = Mm = np.nan

        result["j%d" % j] = dict(
            q=np.array(q_grid), F_s=np.array(fs), grav_resid=np.array(resid),
            slip_plus=np.array(jmp_p), slip_minus=np.array(jmp_m),
            f0_plus=f0p, b_plus=bp, M_plus=Mp, f0_minus=f0m, b_minus=bm, M_minus=Mm,
            f0_kinetic=np.nanmean([f0p, f0m]),
            grav_leak=((f0m - f0p) / 2 if np.isfinite(f0p) and np.isfinite(f0m) else np.nan))
    goto(BASE, secs=3.0)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez(a.out, base=BASE, min_z=MIN_Z, max_x=MAX_X, date="2026-08-23",
         **{"%s_%s" % (jk, kk): v for jk, d in result.items() for kk, v in d.items()})
print("\nsaved %s" % a.out)
print("\n== FRICTION MAP ==")
for j in JOINTS:
    d = result.get("j%d" % j)
    if not d or not len(d["q"]):
        continue
    fs = d["F_s"]
    ok = np.isfinite(fs)
    print("J%d  F_s max/mean/min %.2f/%.2f/%.2f  |  kinetic f0 %.2f N.m (+%.2f/-%.2f)  "
          "b ~%.3f  |  grav leak %.2f  stiction grav-resid rms %.2f" % (
              j + 1, np.nanmax(fs), np.nanmean(fs[ok]), np.nanmin(fs[ok]),
              d["f0_kinetic"], d["f0_plus"], d["f0_minus"],
              np.nanmean([d["b_plus"], d["b_minus"]]), d["grav_leak"],
              np.sqrt(np.nanmean(d["grav_resid"] ** 2))))
print("position hold restored" + (" (%s)" % sess.trip if sess.trip else ""))
