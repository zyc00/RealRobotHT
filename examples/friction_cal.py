"""Friction calibration for the Piper in torque mode - the b601_teleop procedure, ported.

Three measurements per pose, all in t_ff units (what drag mode commands), with the
tool model fed forward so gravity is already cancelled:

  STATIC  (breakaway)  one joint at a time, the rest pinned by the firmware PD. Ramp an
          extra torque u until the joint breaks away, once each way. Breakaway = |qd| above
          0.03 rad/s for a SUSTAINED 40 ms window; the torque recorded is the one at the
          onset of that window (a plain position threshold fires on backlash take-up and
          under-reads - b601 finding).   F_s = (u+ - u-)/2, per-direction u+/u- kept,
          gravity residual = (u+ + u-)/2 (free check of the tool model).
  KINETIC (multi-speed) all joints track a triangle wave at several constant speeds under
          a HOST-side PD sent as t_ff (the firmware kp/kd units are uncalibrated on this
          arm, a host PD is in N.m by construction). The PD torque while sliding is
          friction + model error; splitting by direction and taking the odd part
          (r+ - r-)/2 cancels the model error -> friction(+v) per joint. Several speeds
          -> Coulomb f0 and viscous B.
  LOAD    repeating at spread poses lets `fit` regress f_j(q) = f0_j + mu_j |g_j(q)|.

    python examples/friction_cal.py run                 # auto: visits 6 poses, both sweeps
    python examples/friction_cal.py run --poses 4 --no-static
    python examples/friction_cal.py run --manual        # drag by hand to a pose, then: s / k / d / q
    python examples/friction_cal.py fit                 # -> data/friction_model.npz
    python examples/drag_mode.py --tool data/tool_prior.npz --friction data/friction_model.npz

Rows append to data/friction_cal.csv (kind, pose, per-joint values) so poses accumulate
across runs, exactly like b601's friction.csv. Delete the file to start over.

Runtime (piperx_teleop.FrictionComp): Stribeck two-level curve, static level at the
onset of motion decaying to kinetic (+ load and viscous terms), times --fric-scale
(default 0.8 - the joint must keep meeting SOME real friction or it creeps), times
tanh(v/0.08) so it is zero at rest.
"""
import argparse
import csv
import os
import sys
import threading
import time

import numpy as np

try:
    from piperx_teleop import JOINT_LIMITS, MitCommand, PiperModel, TorqueSession, model_with_tool, require_patched_sdk
    require_patched_sdk()
except (ImportError, RuntimeError):
    _PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

np.set_printoptions(precision=3, suppress=True, linewidth=150)
RAD = np.pi / 180.0
CSV = "data/friction_cal.csv"
OUT = "data/friction_model.npz"
TOOL = "data/tool_prior.npz"

# ---- pinning (firmware PD, uncalibrated units, only used to hold still) ----
PIN_KP, PIN_KD, CATCH_KD = 6.0, 1.0, 3.0
# ---- static breakaway ----
ST_RATE = np.array([0.15, 0.25, 0.15, 0.06, 0.05, 0.04])     # N.m/s ramp
ST_CAP = np.array([1.5, 3.5, 2.0, 1.2, 0.8, 0.5])            # give up beyond this
ST_SHIFT = np.array([0.3, 0.6, 0.3, 0.12, 0.10, 0.08])       # restart shift if it moved the wrong way
ST_KD = 0.3                                                  # light damping on the freed joint
V_ONSET, N_ONSET = 0.03, 8                                   # rad/s, ticks (40 ms at 200 Hz)
RUNWAY = 12.0                                                # deg needed ahead in the ramp direction
# ---- kinetic sweep (host PD in t_ff units) ----
KP_H = np.array([12.0, 20.0, 15.0, 4.0, 3.0, 2.0])           # N.m/rad
KD_H = np.array([0.8, 1.5, 1.0, 0.3, 0.15, 0.10])            # N.m.s/rad
PD_CAP = np.array([1.5, 3.5, 2.5, 1.0, 0.8, 0.5])            # host PD torque clamp
SPEEDS = [0.05, 0.10, 0.20, 0.30]                            # rad/s
PERIOD, AMP_MIN, AMP_MAX, PERIODS = 4.0, 0.12, 0.25, 2       # s, rad, rad, per speed
# Top speed 0.30 with a 0.25 rad amplitude keeps the period >= 3.3 s: J2 (1.1 kg m^2) needs
# ~0.5 s after each reversal to settle, and 0.35 rad/s at a 2.3 s period read 2-3x too high.
# AMP_MIN: the host PD must deflect by breakaway/kp before a joint moves at all (J2: 1.6 N.m /
# 20 N.m/rad = 4.6 deg), so a 0.05 rad/s sweep with a 0.05 rad amplitude never frees J1-J3.
MIN_Z = 0.10
TOOL_LEN = 0.25


class Law:
    """pin | ramp | sweep | drag, switched from the main thread."""

    def __init__(self, mdl):
        self.mdl = mdl
        self.mode = "pin"
        self.q_ref = None
        self.test_j = None
        self.u = 0.0
        self.catch = False
        # sweep state (written by main thread, read here)
        self.sw_center = None
        self.sw_amp = 0.0
        self.sw_period = PERIOD
        self.sw_t0 = None
        self.sw_active = np.ones(6, bool)
        self.lock = threading.Lock()
        self.reset_acc()
        self.last_qdot = np.zeros(6)
        self.qd_f = np.zeros(6)          # EMA of the tick velocity (~20 ms); the raw one is 0 on ticks
        self.QD_ALPHA = 0.25             # where the arm's feedback did not update

    def reset_acc(self):
        self.acc = {+1: [np.zeros(6), np.zeros(6), np.zeros(6)], -1: [np.zeros(6), np.zeros(6), np.zeros(6)]}

    def __call__(self, s):
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        self.qd_f += self.QD_ALPHA * (s.qdot - self.qd_f)
        self.last_qdot = self.qd_f
        g = self.mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        tau = g.copy()
        m = self.mode
        if m == "drag":
            kp[:] = 0.0; kd[:] = 0.0
        elif m == "ramp" and self.test_j is not None:
            j = self.test_j
            tau[j] += self.u
            if self.catch:
                kd[j] = CATCH_KD
            else:
                kp[j] = 0.0; kd[j] = ST_KD
        elif m == "pin" and self.test_j is not None and self.catch:
            kd[self.test_j] = CATCH_KD
        elif m == "sweep":
            if self.sw_t0 is None:
                self.sw_t0 = s.t              # stamped on the session clock at sweep start
            el = s.t - self.sw_t0
            if el < 1.0:
                tri, dtri = 0.0, 0.0
            else:
                ph = ((el - 1.0) % self.sw_period) / self.sw_period
                tri = 4.0 * ph - 1.0 if ph < 0.5 else 3.0 - 4.0 * ph
                dtri = (4.0 / self.sw_period) if ph < 0.5 else (-4.0 / self.sw_period)
            p_des = self.sw_center + self.sw_active * self.sw_amp * tri
            v_des = self.sw_active * self.sw_amp * dtri
            r = np.clip(KP_H * (p_des - s.q) + KD_H * (v_des - self.qd_f), -PD_CAP, PD_CAP)
            tau = g + r
            kp[:] = 0.0; kd[:] = 0.0
            if el >= 1.0:
                frac = ((el - 1.0) % self.sw_period) / self.sw_period
                steady = np.abs(self.qd_f - v_des) < (0.25 * abs(self.sw_amp * 4.0 / self.sw_period) + 0.02)
                d = +1 if 0.15 < frac < 0.45 else (-1 if 0.65 < frac < 0.95 else 0)   # skip 30 % after each reversal
                if d:
                    w = steady.astype(float) * self.sw_active
                    with self.lock:
                        a = self.acc[d]
                        a[0] += w * r; a[1] += w * self.qd_f; a[2] += w
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


def append_rows(rows):
    new = not os.path.exists(CSV)
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["kind", "vset"] + ["q%d" % (i + 1) for i in range(6)]
                       + ["mv%d" % (i + 1) for i in range(6)] + ["f%d" % (i + 1) for i in range(6)]
                       + ["resid%d" % (i + 1) for i in range(6)])
        for kind, vset, q, mv, f, resid in rows:
            w.writerow([kind, "%.5f" % vset] + ["%.5f" % x for x in q] + ["%.5f" % x for x in mv]
                       + ["%.4f" % x for x in f] + ["%.4f" % x for x in resid])


def auto_poses(mdl, n, seed):
    """Spread of shoulder loads and wrist angles, base fixed, tip clear of the table."""
    rng = np.random.default_rng(seed)
    shoulders = [(45, -70), (70, -110), (30, -45), (60, -85), (80, -125), (40, -60)]
    poses = []
    k = 0
    while len(poses) < n and k < 400:
        j2, j3 = shoulders[k % len(shoulders)]; k += 1
        q = np.radians([rng.uniform(-20, 20), j2 + rng.uniform(-5, 5), j3 + rng.uniform(-5, 5),
                        rng.uniform(-45, 45), rng.uniform(-50, 50), rng.uniform(-60, 60)])
        margin = np.degrees(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q)).min()
        if margin < RUNWAY + 15:
            continue
        T, _ = mdl._walk(q)
        tip = T["link6"][:3, 3] + T["link6"][:3, :3] @ [0, 0, TOOL_LEN]
        moving = [l for l in T if l not in ("world", "base_link", "link1", "link2")]   # origins past the shoulder
        if tip[2] < MIN_Z + 0.05 or any(T[l][2, 3] < MIN_Z + 0.05 for l in moving):
            continue
        poses.append(q)
    return poses


def cmd_run(a):
    mdl = model_with_tool(a.tool if os.path.exists(a.tool) else None)
    print("gravity model tool file:", a.tool if os.path.exists(a.tool) else "NONE (bare URDF gripper)")
    law = Law(mdl)
    sess = TorqueSession(law, can=a.can, hz=200.0)
    joints = [j - 1 for j in a.joints]
    links = [k for k in mdl._walk(np.zeros(6))[0] if k not in ("world", "base_link", "link1", "link2")]

    def safe(q):
        T, _ = mdl._walk(q)
        tip = T["link6"][:3, 3] + T["link6"][:3, :3] @ [0, 0, TOOL_LEN]
        return tip[2] > MIN_Z and all(T[k][2, 3] > MIN_Z for k in links)

    def goto(q_t, secs=2.0):
        law.mode = "pin"; law.catch = False; law.u = 0.0
        q0 = law.q_ref.copy() if law.q_ref is not None else sess.q()
        t0 = time.time()
        while time.time() - t0 < secs + 0.3:
            f = min((time.time() - t0) / secs, 1.0)
            law.q_ref = q0 + f * (q_t - q0)
            time.sleep(0.01)

    def ramp(j, u0, direction):
        """Ramp u on freed joint j from u0; sustained-velocity onset detector."""
        q_start = sess.q(); q_pin = q_start[j]
        law.test_j = j; law.u = u0; law.catch = False; law.mode = "ramp"
        t0 = time.time(); cnt = 0; u_onset = np.nan; status = "cap"; u_break = np.nan
        hist = []                                  # (t, q_j) for a 50 ms window velocity
        u_first = np.nan                           # u when the joint first left 0.3 deg (fallback)
        while True:
            time.sleep(0.004)
            now = time.time()
            law.u = u0 + direction * ST_RATE[j] * (now - t0)
            q = sess.q(); dq = (q[j] - q_pin) * direction
            hist.append((now, q[j]))
            while hist and now - hist[0][0] > 0.05:
                hist.pop(0)
            v = (q[j] - hist[0][1]) / max(now - hist[0][0], 1e-3) * direction if len(hist) > 2 else 0.0
            if v > V_ONSET:
                if cnt == 0:
                    u_onset = law.u
                cnt += 1
            else:
                cnt = 0
            if dq > 0.3 * RAD and not np.isfinite(u_first):
                u_first = law.u
            if cnt >= N_ONSET:
                status, u_break = "ok", u_onset; break
            if dq > 3.0 * RAD:                     # moved plenty without a clean velocity onset
                status, u_break = "ok", u_first; break
            if dq < -1.0 * RAD:
                status = "reverse"; break
            if not sess.running:
                status = "tripped"; break
            ahead = (JOINT_LIMITS[j, 1] - q[j]) if direction > 0 else (q[j] - JOINT_LIMITS[j, 0])
            if abs(law.u) >= ST_CAP[j] or ahead < RUNWAY * RAD or not safe(q):
                status = "cap" if abs(law.u) >= ST_CAP[j] else "unsafe"; break
        u_end = law.u
        law.mode = "pin"; law.catch = True; law.q_ref = sess.q()
        time.sleep(0.15)
        law.catch = False; law.u = 0.0
        return u_break, (status if status != "cap" else "cap@%.2f" % u_end)

    def static_sweep(q_pose):
        rows = []
        fs = np.full(6, np.nan); up_all = np.full(6, np.nan); um_all = np.full(6, np.nan); res = np.full(6, np.nan)
        for j in joints:
            ub = {}
            for direction in (+1, -1):
                u0 = 0.0
                for attempt in range(3):
                    goto(q_pose, secs=0.8); time.sleep(0.3)
                    u, status = ramp(j, u0, direction)
                    if status == "ok":
                        ub[direction] = u; break
                    if status == "reverse":
                        u0 += direction * ST_SHIFT[j]; continue
                    if status == "tripped":
                        raise RuntimeError("torque session tripped: %s" % sess.trip)
                    print("    J%d dir %+d: %s (near a limit / in contact?)" % (j + 1, direction, status))
                    break
            if +1 in ub and -1 in ub:
                up_all[j], um_all[j] = ub[1], ub[-1]
                fs[j] = 0.5 * (ub[1] - ub[-1]); res[j] = 0.5 * (ub[1] + ub[-1])
                print("    J%d  u+ %+.3f  u- %+.3f  ->  F_s %.3f   g-resid %+.3f" % (j + 1, ub[1], ub[-1], fs[j], res[j]))
        goto(q_pose, secs=1.0)
        rows.append(("static", 0.0, q_pose, np.zeros(6), fs, res))
        rows.append(("static_pos", 0.0, q_pose, np.zeros(6), up_all, res))
        rows.append(("static_neg", 0.0, q_pose, np.zeros(6), -um_all, res))
        return rows

    def kinetic_sweep(q_pose):
        rows = []
        goto(q_pose, secs=1.0); time.sleep(0.3)
        active = np.zeros(6, bool); active[joints] = True
        def corners_safe(amp, act):
            """Every sign combination of +/-amp on the active joints stays clear of the table."""
            idx = np.where(act)[0]
            for bits in range(1 << len(idx)):
                q = q_pose.copy()
                for b, j in enumerate(idx):
                    q[j] += amp if (bits >> b) & 1 else -amp
                if not safe(q):
                    return False
            return True

        for spd in SPEEDS:
            amp = float(np.clip(spd * PERIOD / 4.0, AMP_MIN, AMP_MAX))
            # runway for the sweep amplitude on every active joint
            ok = np.degrees(np.minimum(q_pose - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q_pose)) > np.degrees(amp) + 5
            act = active & ok
            while amp > 0.6 * AMP_MIN and not corners_safe(amp, act):
                amp *= 0.8                              # low pose: a smaller sweep keeps the tool off the table
            if not corners_safe(amp, act):
                print("    v %.2f rad/s: skipped, no clearance for any sweep amplitude at this pose" % spd)
                continue
            period = 4.0 * amp / spd                    # constant-velocity dwell at exactly spd
            if period < 3.0:                            # heavy joints need ~0.5 s to settle after a reversal
                print("    v %.2f rad/s: skipped, clearance limits the amplitude to %.0f deg -> period %.1f s too short"
                      % (spd, np.degrees(amp), period))
                continue
            nper = PERIODS if period <= 8.0 else 1
            law.sw_active = act
            law.reset_acc()
            law.sw_center = q_pose.copy(); law.sw_amp = amp; law.sw_period = period
            law.sw_t0 = None
            law.mode = "sweep"
            t_end = time.time() + 1.0 + nper * period
            stopped = False
            while time.time() < t_end:
                if not sess.running:
                    raise RuntimeError("torque session tripped: %s" % sess.trip)
                if not safe(sess.q()):
                    print("    v %.2f rad/s: sweep stopped by the clearance guard, no data" % spd); stopped = True; break
                time.sleep(0.02)
            with law.lock:
                acc = {d: [x.copy() for x in law.acc[d]] for d in (+1, -1)}
            law.mode = "pin"; law.q_ref = sess.q()
            fp = acc[1][0] / np.maximum(acc[1][2], 1); vp = acc[1][1] / np.maximum(acc[1][2], 1)
            fn = acc[-1][0] / np.maximum(acc[-1][2], 1); vn = acc[-1][1] / np.maximum(acc[-1][2], 1)
            fp[acc[1][2] < 10] = np.nan; fn[acc[-1][2] < 10] = np.nan
            stuck = law.sw_active & (np.abs(0.5 * (vp - vn)) < 0.3 * spd)
            if stuck.any() and not stopped:
                print("    (J%s did not follow the %.2f rad/s sweep: PD deflection below stiction at this amplitude)"
                      % (",".join(str(j + 1) for j in np.where(stuck)[0]), spd))
            odd = 0.5 * (fp - fn); even = 0.5 * (fp + fn)
            if stopped:
                time.sleep(0.3); continue
            print("    v %.2f rad/s (amp %.0f deg): |v|meas %s  friction(odd) %s  model-resid(even) %s" %
                  (spd, np.degrees(amp), (0.5 * (vp - vn)).round(3), odd.round(3), even.round(3)))
            rows.append(("kin", +spd, q_pose, vp, fp, even))
            rows.append(("kin", -spd, q_pose, vn, fn, even))
            time.sleep(0.3)
        goto(q_pose, secs=1.0)
        return rows

    if a.manual:
        print("MANUAL: the arm goes compliant (drag). Move it to a pose, let go, then type a key + Enter:")
        print("   s = static breakaway sweep   k = kinetic multi-speed sweep   d = back to drag   q = quit")
    else:
        poses = auto_poses(PiperModel(), a.poses, a.seed)
        print("AUTO: %d poses, joints %s, static %s, kinetic %s" %
              (len(poses), [j + 1 for j in joints], not a.no_static, not a.no_kinetic))
    print("!! MIT mode. Workspace clear, hand on the E-STOP. Hands off during sweeps.")
    if not a.yes:
        input(">>> ENTER to start ")

    all_rows = []
    try:
        with sess:
            if a.manual:
                law.mode = "drag"
                while True:
                    c = input("> ").strip().lower()
                    if c == "q":
                        break
                    if c == "d":
                        law.mode = "drag"; continue
                    if c in ("s", "k"):
                        q_pose = sess.q().copy()
                        law.q_ref = q_pose.copy(); law.mode = "pin"; time.sleep(0.5)
                        rows = static_sweep(q_pose) if c == "s" else kinetic_sweep(q_pose)
                        append_rows(rows); all_rows += rows
                        print("  appended %d rows to %s; 'd' to drag on" % (len(rows), CSV))
                        law.mode = "drag"
            else:
                for k, q_pose in enumerate(poses):
                    print("pose %d/%d  %s deg" % (k + 1, len(poses), np.degrees(q_pose).round(0)))
                    goto(q_pose, secs=3.0); time.sleep(0.5)
                    rows = []
                    if not a.no_static:
                        rows += static_sweep(q_pose)
                    if not a.no_kinetic:
                        rows += kinetic_sweep(q_pose)
                    append_rows(rows); all_rows += rows
                goto(poses[0], secs=3.0)
    except KeyboardInterrupt:
        print("\ninterrupted")
    except RuntimeError as e:
        print("\nABORTED:", e)
    if sess.trip:
        print("session tripped:", sess.trip)
    print("%d rows appended to %s   ->  python examples/friction_cal.py fit" % (len(all_rows), CSV))


def _robust_fit(A, y):
    w = np.ones(len(y)); coef = np.zeros(A.shape[1])
    for _ in range(12):
        W = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
        res = y - A @ coef
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        aa = np.abs(res)
        w = np.where(aa <= 1.345 * s, 1.0, 1.345 * s / np.maximum(aa, 1e-9))
    return coef, float(np.sqrt(np.mean((y - A @ coef) ** 2)))


def cmd_fit(a):
    mdl = model_with_tool(a.tool if os.path.exists(a.tool) else None)
    rows = list(csv.DictReader(open(CSV, newline="")))
    Q = lambda r: np.array([float(r["q%d" % (i + 1)]) for i in range(6)])
    F = lambda r, key: np.array([float(r[key % (i + 1)]) for i in range(6)])
    gabs = lambda q: np.abs(mdl.gravity_torque(q))
    out = dict(kin_f0=np.full(6, np.nan), kin_mu=np.zeros(6), kin_B=np.zeros(6),
               static_f0=np.full(6, np.nan), static_mu=np.zeros(6),
               static_pos=np.full(6, np.nan), static_neg=np.full(6, np.nan),
               kin_rms=np.zeros(6), static_spread=np.zeros(6))    # uncertainty -> passivity bound (eq 35)

    # ---- kinetic: pair +/-v per pose, odd part vs measured speed --------------
    kin = [r for r in rows if r["kind"] == "kin"]
    plus, minus = {}, {}
    for r in kin:
        key = (tuple(np.round(Q(r), 4)), round(abs(float(r["vset"])), 5))
        (plus if float(r["vset"]) > 0 else minus)[key] = (F(r, "mv%d"), F(r, "f%d"))
    pairs = sorted(set(plus) & set(minus))
    V, ODD, G = [], [], []
    for key in pairs:
        mvp, fp = plus[key]; mvn, fn = minus[key]
        V.append(0.5 * (mvp - mvn)); ODD.append(0.5 * (fp - fn)); G.append(gabs(np.array(key[0])))
    V, ODD, G = (np.array(x) if x else np.zeros((0, 6)) for x in (V, ODD, G))
    print("kinetic: %d (+v,-v) pairs from %d poses" % (len(pairs), len({k[0] for k in pairs})))
    print("  joint   f0     B(N.m.s)   mu(load)   rms flat -> load   gspan")
    for j in range(6):
        ok = np.isfinite(ODD[:, j]) & np.isfinite(V[:, j]) & (V[:, j] > 1e-3) if len(V) else np.zeros(0, bool)
        if ok.sum() < 3:
            print("  J%d     (not enough data)" % (j + 1)); continue
        v, y, g = V[ok, j], ODD[ok, j], G[ok, j]
        (f0, B), rms_flat = _robust_fit(np.c_[np.ones_like(v), v], y)
        mu = 0.0; rms = rms_flat; gspan = g.max() - g.min()
        if ok.sum() >= 6 and gspan >= 0.4:
            (f0l, mul, Bl), rms_l = _robust_fit(np.c_[np.ones_like(v), g, v], y)
            if mul > 0 and rms_l < 0.9 * rms_flat:
                f0, B, mu, rms = f0l, Bl, mul, rms_l
        B = max(B, 0.0)
        f0 = f0 - rms                                    # conservative: below the real level at every pose
        out["kin_f0"][j], out["kin_B"][j], out["kin_mu"][j] = f0, B, mu
        out["kin_rms"][j] = rms
        print("  J%d   %6.3f   %7.4f    %6.3f     %.3f -> %.3f    %.2f" % (j + 1, f0, B, mu, rms_flat, rms, gspan))

    # ---- static: level (+ load), per-direction medians -----------------------
    def fit_kind(kind):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            return np.full(6, np.nan), np.zeros(6), 0
        Fm = np.array([F(r, "f%d") for r in rs]); Gm = np.array([gabs(Q(r)) for r in rs])
        # CONSERVATIVE level: the compensation must stay below the real friction at EVERY pose
        # (measured pose-to-pose spread on the wrist is 2-3x, far more than the 20 % scale margin),
        # so without a load model take the 25th percentile, with one take the fit minus its rms.
        f0 = np.nanmin(Fm, axis=0); mu = np.zeros(6)          # static: the MIN over poses (onset level is the risky one)
        for j in range(6):
            ok = np.isfinite(Fm[:, j])
            if ok.sum() >= 4 and (Gm[ok, j].max() - Gm[ok, j].min()) >= 0.4:
                (c0, m), rms_l = _robust_fit(np.c_[np.ones(ok.sum()), Gm[ok, j]], Fm[ok, j])
                rms_med = np.sqrt(np.mean((Fm[ok, j] - np.nanmedian(Fm[ok, j])) ** 2))
                if m > 0 and rms_l < 0.8 * rms_med:
                    f0[j], mu[j] = c0 - rms_l, m
        return f0, mu, len(rs)

    out["static_f0"], out["static_mu"], n_st = fit_kind("static")
    st_rows_all = [r for r in rows if r["kind"] == "static"]
    # per-direction levels are SYMMETRIC on purpose: for a Coulomb model, direction-asymmetric
    # friction (F+ != F-) is identical to symmetric friction (F+ + F-)/2 plus a constant torque
    # -(F+ - F-)/2, and that constant is exactly the (u+ + u-)/2 "gravity residual" column, which
    # examples/gravity_cal.py fits as a per-joint bias. Splitting here as well would count it twice.
    out["static_pos"] = out["static_f0"].copy(); out["static_neg"] = out["static_f0"].copy()
    st_rows = [r for r in rows if r["kind"] == "static"]
    if st_rows:
        Fm = np.array([F(r, "f%d") for r in st_rows]); Rm = np.array([F(r, "resid%d") for r in st_rows])
        out["static_spread"] = np.nanmax(Fm, 0) - np.nanmin(Fm, 0)
        print("\nstatic breakaway: %d poses" % n_st)
        print("  joint   F_s med   min    max    mu(load)   +dir    -dir   |g-resid| mean   (levels saved = conservative)")
        for j in range(6):
            print("  J%d     %6.3f  %6.3f %6.3f   %6.3f    %6.3f  %6.3f   %.3f" % (
                j + 1, np.nanmedian(Fm[:, j]), np.nanmin(Fm[:, j]), np.nanmax(Fm[:, j]), out["static_mu"][j],
                out["static_pos"][j], out["static_neg"][j], np.nanmean(np.abs(Rm[:, j]))))
        print("  (a |g-resid| well above F_s/3 on a joint means the tool/gravity model is off there)")
    # runtime falls back: static -> kinetic where missing, kinetic -> static
    for j in range(6):
        if not np.isfinite(out["kin_f0"][j]) and np.isfinite(out["static_f0"][j]):
            out["kin_f0"][j] = out["static_f0"][j]
        if not np.isfinite(out["static_f0"][j]) and np.isfinite(out["kin_f0"][j]):
            out["static_f0"][j] = out["kin_f0"][j]
        for k in ("static_pos", "static_neg"):
            if not np.isfinite(out[k][j]):
                out[k][j] = out["static_f0"][j]
    np.savez(OUT, **out, tool=a.tool, csv=CSV)
    print("\nsaved %s\n   ->  python examples/drag_mode.py --tool %s --friction %s" % (OUT, a.tool, OUT))


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("run"); p.set_defaults(f=cmd_run)
p.add_argument("--manual", action="store_true", help="drag to poses by hand, keys s/k/d/q")
p.add_argument("--poses", type=int, default=6)
p.add_argument("--seed", type=int, default=5)
p.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
p.add_argument("--no-static", action="store_true")
p.add_argument("--no-kinetic", action="store_true")
p.add_argument("--tool", default=TOOL)
p.add_argument("--can", default="can0")
p.add_argument("--yes", action="store_true")
p = sub.add_parser("fit"); p.set_defaults(f=cmd_fit)
p.add_argument("--tool", default=TOOL)
a = ap.parse_args()
a.f(a)
