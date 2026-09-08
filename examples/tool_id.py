"""Identify the tool past joint 6 (adaptor + Nano25 F/T + adaptor + gripper).

Everything bolted past joint 6 is ONE rigid body, so gravity only needs its
mass and centre of mass in the link6 (flange) frame: 4 barycentric parameters
beta = (m, m*cx, m*cy, m*cz). link6 +z is the tool axis, out of the flange.

    python examples/tool_id.py prior      # CAD + datasheet estimate -> data/tool_prior.npz
    python examples/tool_id.py torque     # MIT-mode breakaway residuals -> data/tool_torque.npz
    python examples/tool_id.py fit        # -> data/tool_body.npz
    python examples/drag_mode.py --tool data/tool_body.npz     # the real test

torque (the measurement that counts): the arm is pinned in MIT mode with the
PRIOR tool fed forward. At each pose one joint is freed and an extra torque u
is ramped until the joint breaks away, once upward and once downward:

    u+ = (true_g - model) + F_s        u- = (true_g - model) - F_s
    residual = (u+ + u-)/2   in t_ff units, friction cancelled exactly

The ramps start from a PD-deflection estimate of the residual (the pin's kp
is calibrated in place by adding a known torque), so they are short and the
joint barely moves (catch at BREAK_DEG). Wrist joints J4/J5 carry the
information; J3 is optional and slower (friction ~1 N.m).

fit: residual_j(q) = Y_j(q) @ dbeta, ridge toward 0, beta = beta_prior + dbeta.
Since t_ff IS the unit drag mode runs in, the fitted mass is a real kg figure.

collect/check (position mode, reported effort): the effort sensor confirms the
model's SHAPE per joint (correlation) but CANNOT weigh the tool - each joint
reports current with its own unknown coefficient (measured 0.28/0.29/0.82/1.16
on J2-J5), so mass and coefficient trade off. Use it as a plausibility check
only.
"""
import argparse
import os
import struct
import sys
import time

import numpy as np

try:
    from piperx_teleop import JOINT_LIMITS, PiperArm, PiperModel
    if len(sys.argv) > 1 and sys.argv[1] == "torque":
        from piperx_teleop import MitCommand, TorqueSession, require_patched_sdk
        require_patched_sdk()
except (ImportError, RuntimeError):
    _PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

np.set_printoptions(precision=4, suppress=True)
RAD = np.pi / 180.0
LINK = "link6"
DATA = "data/tool_id.npz"
OUT = "data/tool_body.npz"
TORQUE = "data/tool_torque.npz"
PRIOR = "data/tool_prior.npz"

# --- the stack, arm side -> tool side, along link6 +z --------------------------
STL_ARM = os.path.expanduser("~/Downloads/p_arm_v3.stl")          # 12 mm disc
STL_TOOL = os.path.expanduser("~/Downloads/px_tool_noflange.stl")  # 15 mm disc
NANO25_MASS = 0.0635         # kg, ATI spec (drawing's weight field is blank)
NANO25_H = 0.0216            # m, mounting face to tool face (drawing: 21.6)
# ATI: "Connector (not shown) has 17 mm diameter and is 67.5 mm long" - radial,
# unmodelled here. If the fit's cx/cy come out a few mm off-axis, that is it.

# reported-effort scatter while holding still, per joint (friction band, N.m):
# weights the joint fit so the quiet wrist joints count more than J2/J3
SIGMA = np.array([0.30, 0.60, 0.30, 0.09, 0.07, 0.05])


# --- prior from CAD ------------------------------------------------------------
def stl_volume_centroid(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    d = np.frombuffer(b[84:84 + n * 50],
                      dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    T = d["v"].astype(float) * 1e-3                    # mm -> m
    a, b3, c = T[:, 0], T[:, 1], T[:, 2]
    v6 = np.einsum("ij,ij->i", a, np.cross(b3, c))
    V = v6.sum() / 6
    cen = ((a + b3 + c) / 4 * v6[:, None]).sum(0) / v6.sum()
    z0, z1 = T[:, :, 2].min(), T[:, :, 2].max()
    return V, cen, z0, z1


def prior(density):
    """(mass, com in link6 frame) of adaptor + sensor + adaptor + gripper."""
    mdl = PiperModel()
    parts = []           # (name, m, com)
    z = 0.0
    V, cen, z0, z1 = stl_volume_centroid(STL_ARM)
    parts.append(("arm adaptor", V * density, np.array([cen[0], cen[1], z + cen[2] - z0])))
    z += z1 - z0
    parts.append(("Nano25", NANO25_MASS, np.array([0.0, 0.0, z + NANO25_H / 2])))
    z += NANO25_H
    V, cen, z0, z1 = stl_volume_centroid(STL_TOOL)
    parts.append(("tool adaptor", V * density, np.array([cen[0], cen[1], z + cen[2] - z0])))
    z += z1 - z0
    # gripper (URDF gripper_base + fingers, its own flange plate dropped) shifted
    # so gripper_base's origin sits at the top of the stack instead of +4.5 mm
    m6, c6 = mdl.links["link6"]
    parts.append(("link6 (URDF)", m6, c6))
    T6, _ = mdl._walk(np.zeros(6))
    Tinv = np.linalg.inv(T6["link6"])
    for l in ("gripper_base", "gripper_link1", "gripper_link2"):
        m, c = mdl.links[l]
        p = (Tinv @ T6[l] @ np.append(c, 1.0))[:3]
        p[2] += z - 0.0045
        parts.append((l + " (URDF, shifted)", m, p))
    m_tot = sum(m for _, m, _ in parts)
    com = sum(m * c for _, m, c in parts) / m_tot
    return m_tot, com, parts, z


def cmd_prior(a):
    m, com, parts, z = prior(a.density)
    print("stack height flange -> gripper base: %.1f mm" % (z * 1e3))
    print("%-28s %8s   %s" % ("part", "mass g", "com in link6 (mm)"))
    for name, mp, c in parts:
        print("%-28s %8.1f   %s" % (name, mp * 1e3, (c * 1e3).round(1)))
    print("%-28s %8.1f   %s" % ("TOTAL", m * 1e3, (com * 1e3).round(1)))
    m0, c0 = PiperModel().distal_body_urdf()
    print("\nURDF distal body (old gripper only): %.1f g at %s mm" % (m0 * 1e3, (c0 * 1e3).round(1)))
    ma, coma, _, _ = prior(2.70e3)
    print("same stack if adaptors are aluminium: %.1f g at %s mm" % (ma * 1e3, (coma * 1e3).round(1)))
    np.savez(PRIOR, mass=m, com=com, link=LINK, density=a.density)
    print("saved", PRIOR)


# --- collect ---------------------------------------------------------------------
def build_poses(mdl, n, seed, tool_len, wrist=75):
    """Spread of wrist orientations over a few shoulder configs, base still."""
    rng = np.random.default_rng(seed)
    shoulders = [(45, -70), (65, -100), (30, -45), (80, -120)]
    poses = []
    tries = 0
    while len(poses) < n and tries < 5000:
        tries += 1
        j2, j3 = shoulders[len(poses) % len(shoulders)]
        q = np.radians([0.0, j2 + rng.uniform(-5, 5), j3 + rng.uniform(-5, 5),
                        rng.choice([-wrist, -wrist / 2, 0, wrist / 2, wrist]) + rng.uniform(-5, 5),
                        rng.choice([-wrist, -wrist / 2, 0, wrist / 2, wrist]) + rng.uniform(-5, 5),
                        rng.choice([-90, -45, 0, 45, 90]) + rng.uniform(-5, 5)])
        margin = np.degrees(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q)).min()
        if margin < 10:
            continue
        T = mdl.fk(q, LINK)
        tip = T[:3, 3] + T[:3, :3] @ np.array([0, 0, tool_len])
        for p in (T[:3, 3], tip):
            if p[2] < 0.15 or np.hypot(p[0], p[1]) > 0.55:
                break
        else:
            poses.append(q)
    order, rem = [0], list(range(1, len(poses)))
    while rem:
        last = poses[order[-1]]
        nxt = min(rem, key=lambda i: np.abs(poses[i] - last).sum())
        order.append(nxt); rem.remove(nxt)
    return [poses[i] for i in order]


def sample(arm, q, secs, hz=50):
    Q, E = [], []
    t0 = time.time()
    while time.time() - t0 < secs:
        arm.move_j(q, speed_pct=10)
        Q.append(arm.q()); E.append(arm.effort())
        time.sleep(1.0 / hz)
    return np.mean(Q, 0), np.mean(E, 0), np.std(E, 0)


def cmd_collect(a):
    mdl = PiperModel()
    poses = build_poses(mdl, a.poses, a.seed, a.tool_len)
    poses += poses[:a.repeat]                        # revisits -> repeatability
    print("%d holds (%d poses + %d revisits), 2 approaches each" % (len(poses), a.poses, a.repeat))
    print("!! The arm will move through the wrist range. Clear the workspace.")
    print("!! DO NOT TOUCH IT - every sample is labelled as pure gravity.")
    if not a.yes:
        input(">>> ENTER to start ")

    arm = PiperArm(a.can).connect(0.5)
    if not all(arm.is_enabled()):
        q = arm.q(); arm.piper.EnableArm(7)
        t0 = time.time()
        while time.time() - t0 < 2.0:
            arm.move_j(q, speed_pct=10); time.sleep(0.01)

    delta = np.radians([0, a.delta, a.delta, a.delta, a.delta, a.delta])
    Q, E, S, POSE, SIDE = [], [], [], [], []
    t_begin = time.time()
    try:
        for k, q in enumerate(poses):
            for side in (-1, +1):
                if not arm.move_to(q + side * delta, speed_pct=a.speed, tol_deg=1.5, timeout=25):
                    print("  pose %d: approach timed out, skipping" % k); continue
                time.sleep(0.3)
                arm.move_to(q, speed_pct=a.speed, tol_deg=1.0, timeout=15)
                time.sleep(a.settle)
                qm, em, es = sample(arm, q, a.sample)
                Q.append(qm); E.append(em); S.append(es); POSE.append(k); SIDE.append(side)
            print("  %2d/%d  q=%s  effort=%s  (%.0f s)" %
                  (k + 1, len(poses), np.degrees(q).round(0), em.round(2), time.time() - t_begin))
    except KeyboardInterrupt:
        print("\ninterrupted - saving what we have")
    finally:
        arm.hold(0.5)
    np.savez(DATA, q=np.array(Q), effort=np.array(E), effort_std=np.array(S),
             pose=np.array(POSE), side=np.array(SIDE))
    print("saved %s  (%d holds)" % (DATA, len(Q)))


# --- fit ------------------------------------------------------------------------
def cmd_check(a):
    """Per-joint regression of reported effort on the model with a tool file."""
    d = np.load(DATA)
    Q, E = d["q"], d["effort"]
    from piperx_teleop import model_with_tool
    mdl = model_with_tool(a.tool if os.path.exists(a.tool) else None)
    G = np.array([mdl.gravity_torque(q) for q in Q])
    print("effort = slope * G_model + offset, per joint   (tool: %s, %d holds)" % (a.tool, len(Q)))
    print("  joint   slope   offset   corr    rms   | slope is the effort coefficient, NOT a mass error")
    for j in range(6):
        g, e = G[:, j], E[:, j]
        if g.std() < 1e-6:
            print("  J%d      no gravity signal" % (j + 1)); continue
        sl, off = np.polyfit(g, e, 1)
        r = np.corrcoef(g, e)[0, 1]
        print("  J%d    %6.3f  %7.3f  %6.3f  %6.3f" % (j + 1, sl, off, r, np.std(e - (sl * g + off))))
    print("corr < 0.95 on J2-J5 means the model SHAPE is wrong; the mass cannot be read from this")


# --- torque-mode residuals ------------------------------------------------------
PIN_KP, PIN_KD = 6.0, 1.0        # firmware PD, units uncalibrated - only used to pin
CATCH_KD = 3.0
RAMP_KD = 0.3                    # light damping on the freed joint
BREAK_DEG = 0.6
JOINT_PLAN = {                   # j0 -> (ramp rate N.m/s, calibration torque, cap, retry shift)
    2: (0.20, 1.00, 3.0, 0.50),
    3: (0.08, 0.20, 1.5, 0.15),
    4: (0.06, 0.15, 1.2, 0.12),
}
MIN_Z_TORQUE = 0.10
RUNWAY_DEG = 12.0


class PinLaw:
    def __init__(self, mdl):
        self.mdl = mdl
        self.q_ref = None
        self.test_j = None
        self.mode = "pin"           # pin | ramp
        self.u = 0.0                # extra torque on test_j (both modes)
        self.catch = False

    def __call__(self, s):
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        tau = self.mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        j = self.test_j
        if j is not None:
            tau[j] += self.u
            if self.catch:
                kd[j] = CATCH_KD
            elif self.mode == "ramp":
                kp[j] = 0.0
                kd[j] = RAMP_KD
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


def cmd_torque(a):
    p = np.load(a.prior)
    m_p, c_p = float(p["mass"]), np.asarray(p["com"], float)
    mdl = PiperModel(distal_body=(m_p, c_p))
    joints = [j - 1 for j in a.joints]
    poses = build_poses(PiperModel(), a.poses, a.seed, a.tool_len, wrist=60)
    # every test joint needs runway both ways at every pose
    poses = [q for q in poses if all(
        np.degrees(min(q[j] - JOINT_LIMITS[j, 0], JOINT_LIMITS[j, 1] - q[j])) > RUNWAY_DEG + 6 for j in joints)]
    print("%d poses x joints %s  (prior %.0f g at %s mm)" %
          (len(poses), [j + 1 for j in joints], m_p * 1e3, (c_p * 1e3).round(0)))
    print("!! MIT mode. The arm pins itself, then frees one joint at a time and nudges it")
    print("!! ~%.1f deg. Clear the workspace, hand on the E-STOP, do not touch the arm." % BREAK_DEG)
    if not a.yes:
        input(">>> ENTER to start ")

    law = PinLaw(mdl)
    sess = TorqueSession(law, can=a.can, hz=200.0)
    links = [k for k in mdl._walk(np.zeros(6))[0] if k not in ("world", "base_link")]

    def safe(q):
        T, _ = mdl._walk(q)
        tip = T[LINK][:3, 3] + T[LINK][:3, :3] @ [0, 0, a.tool_len]
        return tip[2] > MIN_Z_TORQUE and all(T[k][2, 3] > MIN_Z_TORQUE for k in links)

    def goto(q_t, secs=2.0):
        law.mode = "pin"; law.catch = False; law.u = 0.0
        q0 = law.q_ref.copy() if law.q_ref is not None else sess.q()
        t0 = time.time()
        while time.time() - t0 < secs + 0.3:
            f = min((time.time() - t0) / secs, 1.0)
            law.q_ref = q0 + f * (q_t - q0)
            time.sleep(0.01)

    def deflection(j, secs=0.6):
        t0 = time.time(); d = []
        while time.time() - t0 < secs:
            d.append(law.q_ref[j] - sess.q()[j]); time.sleep(0.01)
        return float(np.mean(d))

    def pd_estimate(j, u_cal):
        """Calibrate the pin's kp on joint j in place; return (residual_est, kp_true)."""
        law.test_j = j; law.mode = "pin"; law.u = 0.0
        time.sleep(0.8); d0 = deflection(j)
        law.u = u_cal
        time.sleep(1.0); d1 = deflection(j)
        law.u = 0.0
        time.sleep(0.5)
        dd = d0 - d1
        if abs(dd) < 1e-4:
            return np.nan, np.nan
        kp = u_cal / dd
        return kp * d0, kp          # PD torque kp*(q_ref-q) balances the residual

    def ramp(j, u0, direction, rate, cap):
        """From u0, ramp u in `direction` until the joint moves BREAK_DEG that way.
        Returns (u_break, 'ok'|'reverse'|'cap'|'unsafe')."""
        q_start = sess.q()
        q_pin = q_start[j]
        law.test_j = j; law.u = u0; law.catch = False; law.mode = "ramp"
        t0 = time.time(); result = ("cap", np.nan)
        while True:
            time.sleep(0.004)
            law.u = u0 + direction * rate * (time.time() - t0)
            q = sess.q()
            dq = (q[j] - q_pin) * direction
            if dq > BREAK_DEG * RAD:
                result = ("ok", law.u); break
            if dq < -BREAK_DEG * RAD:
                result = ("reverse", law.u); break
            ahead = (JOINT_LIMITS[j, 1] - q[j]) if direction > 0 else (q[j] - JOINT_LIMITS[j, 0])
            if abs(law.u) >= cap or ahead < RUNWAY_DEG * RAD or not safe(q) or not sess.running:
                result = ("cap" if abs(law.u) >= cap else "unsafe", np.nan); break
        law.mode = "pin"; law.catch = True; law.q_ref = sess.q()
        time.sleep(0.15)
        law.catch = False; law.u = 0.0
        return result[1], result[0], q_start

    rows = []   # (joint, q, residual, F_s, u_plus, u_minus, pd_est, kp)
    t_begin = time.time()
    try:
        with sess:
            for k, q_pose in enumerate(poses):
                goto(q_pose, secs=2.5); time.sleep(0.4)
                for j in joints:
                    rate, u_cal, cap, shift = JOINT_PLAN[j]
                    est, kp = pd_estimate(j, u_cal)
                    if not np.isfinite(est):
                        print("  pose %d J%d: pin did not deflect, skipping" % (k, j + 1)); continue
                    est = float(np.clip(est, -cap / 2, cap / 2))
                    ub, q_at = {}, q_pose
                    for direction in (+1, -1):
                        u0 = est
                        for attempt in range(3):
                            goto(q_pose, secs=0.8); time.sleep(0.3)
                            u, status, q_start = ramp(j, u0, direction, rate, cap)
                            if status == "ok":
                                ub[direction] = u; q_at = q_start; break
                            if status == "reverse":
                                u0 += direction * shift        # started outside the band
                                continue
                            break                              # cap / unsafe: give up
                    if +1 in ub and -1 in ub:
                        res = 0.5 * (ub[1] + ub[-1]); fs = 0.5 * (ub[1] - ub[-1])
                        rows.append((j, q_at, res, fs, ub[1], ub[-1], est, kp))   # q as actually pinned
                        print("  %2d/%d J%d  pd_est %+.3f  u+ %+.3f  u- %+.3f  ->  residual %+.3f  F_s %.3f  (%.0f s)" %
                              (k + 1, len(poses), j + 1, est, ub[1], ub[-1], res, fs, time.time() - t_begin))
                    else:
                        print("  %2d/%d J%d  incomplete %s" % (k + 1, len(poses), j + 1, ub))
                    goto(q_pose, secs=1.0)
            goto(poses[0], secs=2.5)
    except KeyboardInterrupt:
        print("\ninterrupted - saving what we have")
    if sess.trip:
        print("session tripped:", sess.trip)
    if rows:
        np.savez(TORQUE, joint=np.array([r[0] for r in rows]), q=np.array([r[1] for r in rows]),
                 residual=np.array([r[2] for r in rows]), F_s=np.array([r[3] for r in rows]),
                 u_plus=np.array([r[4] for r in rows]), u_minus=np.array([r[5] for r in rows]),
                 pd_est=np.array([r[6] for r in rows]), kp=np.array([r[7] for r in rows]),
                 prior_mass=m_p, prior_com=c_p)
        print("saved %s  (%d residuals)   ->  python examples/tool_id.py fit" % (TORQUE, len(rows)))


def cmd_fit(a):
    d = np.load(TORQUE)
    J, Q, R, FS = d["joint"], d["q"], d["residual"], d["F_s"]
    m_p, c_p = float(d["prior_mass"]), np.asarray(d["prior_com"], float)
    beta_p = np.concatenate([[m_p], m_p * c_p])
    mdl = PiperModel()
    Y = np.array([mdl.gravity_regressor(q, [LINK])[j] for j, q in zip(J, Q)])     # (N,4)
    w = 1.0 / np.maximum(FS, 0.02)                                                # quiet joints count more

    def solve(idx):
        A = Y[idx] * w[idx, None]; y = R[idx] * w[idx]
        P = np.eye(4) * a.lam * len(idx)
        return np.linalg.solve(A.T @ A + P, A.T @ y)

    idx = np.arange(len(R))
    db = solve(idx)
    beta = beta_p + db
    m = beta[0]; com = beta[1:] / m
    pred = Y @ db
    print("%d residuals: " % len(R) + "  ".join("J%d x%d" % (j + 1, (J == j).sum()) for j in np.unique(J)))
    print("prior : m=%.1f g  com=%s mm" % (m_p * 1e3, (c_p * 1e3).round(1)))
    print("fitted: m=%.1f g  com=%s mm    (t_ff units = kg, the units drag mode runs in)" % (m * 1e3, (com * 1e3).round(1)))
    print("\nrms residual torque per joint (N.m), before -> after -> leave-one-out")
    loo = np.array([R[i] - Y[i] @ solve(np.delete(idx, i)) for i in idx])
    for j in np.unique(J):
        s = J == j
        print("  J%d   %.3f -> %.3f -> %.3f     (F_s mean %.3f)" %
              (j + 1, np.sqrt(np.mean(R[s] ** 2)), np.sqrt(np.mean((R[s] - pred[s]) ** 2)),
               np.sqrt(np.mean(loo[s] ** 2)), FS[s].mean()))
    sv = np.linalg.svd(Y * w[:, None], compute_uv=False)
    print("\nregressor singular values %s  (ratio %.0f; > 300 means some direction is poorly excited)" %
          (sv.round(2), sv[0] / max(sv[-1], 1e-9)))
    if m <= 0.1 or m > 2.0 or np.linalg.norm(com - c_p) > 0.06:
        print("!! implausible body - inspect the residual table before trusting this file")
    np.savez(OUT, mass=m, com=com, link=LINK, beta=beta, dbeta=db, prior_mass=m_p, prior_com=c_p,
             rms_loo=np.sqrt(np.mean(loo ** 2)))
    print("\nsaved %s   ->  python examples/drag_mode.py --tool %s" % (OUT, OUT))


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("prior", help="CAD + datasheet estimate"); p.set_defaults(f=cmd_prior)
p.add_argument("--density", type=float, default=1.24e3, help="adaptor material kg/m^3 (PLA 1240, Al 2700)")
p = sub.add_parser("torque", help="MIT-mode breakaway residuals (the real measurement)"); p.set_defaults(f=cmd_torque)
p.add_argument("--joints", type=int, nargs="+", default=[4, 5], help="1-indexed; add 3 for a slow mass check")
p.add_argument("--poses", type=int, default=12)
p.add_argument("--seed", type=int, default=3)
p.add_argument("--tool-len", type=float, default=0.25)
p.add_argument("--prior", default=PRIOR)
p.add_argument("--can", default="can0")
p.add_argument("--yes", action="store_true")
p = sub.add_parser("fit", help="fit mass+COM to the torque residuals"); p.set_defaults(f=cmd_fit)
p.add_argument("--lam", type=float, default=1e-3, help="ridge toward the prior")
p = sub.add_parser("collect", help="position-mode effort survey (shape check only)"); p.set_defaults(f=cmd_collect)
p.add_argument("--poses", type=int, default=24)
p.add_argument("--repeat", type=int, default=4, help="poses revisited at the end")
p.add_argument("--seed", type=int, default=3)
p.add_argument("--speed", type=int, default=15)
p.add_argument("--settle", type=float, default=1.0)
p.add_argument("--sample", type=float, default=0.8)
p.add_argument("--delta", type=float, default=3.0, help="deg approach offset each side")
p.add_argument("--tool-len", type=float, default=0.25, help="m past the flange kept above the table")
p.add_argument("--can", default="can0")
p.add_argument("--yes", action="store_true")
p = sub.add_parser("check", help="effort-vs-model shape check for a tool file"); p.set_defaults(f=cmd_check)
p.add_argument("--tool", default=OUT)
a = ap.parse_args()
a.f(a)
