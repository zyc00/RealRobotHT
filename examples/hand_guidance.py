"""Hand-guidance compliance for a non-uniform gear-ratio arm.

Faithful implementation of the two-layer design (doc: "virtual friction
uniformization + low-gear-ratio joint force observation"):

LAYER 1 - per-joint virtual friction uniformization, no cross coupling:
    tau_i = g_i(q) + alpha_i * f_hat_i(qdot_i)   (H = J1-J3: cancel friction)
                   - b_v,i * qdot_i              (L = J4-J6: inject damping)
    alpha_i = 1 - tau_s_v / tau_s_i^eff, capped at ALPHA_MAX (model-mismatch
    margin). L gets NO virtual Coulomb (would re-create a dead zone); its
    damping doubles as Layer 2's sensor.

LAYER 2 - the wrist as a force observer (solves breakaway):
    quasi-static on damped L:   tau_ext,L ~= B_L qdot_L
    rank-3 estimate:            F_hat = (J_vL^T)^{+lambda} B_L qdot_L
    feed-forward to H:          tau_a,H = kappa * J_vH^T F_hat
    breakaway event (stuck i, |tau_ext_i| > delta): inject the UPPER-BOUND
    stiction along sgn(tau_ext_i); a stick-slip state machine ramps it down
    to the Coulomb level within ~30 ms of first motion.

    Structural anti-self-drive: the observer reads ONLY L velocities - H
    motion cannot feed its own assist (the failure measured on 08-21 when a
    full-xdot deficit assist walked the arm to J2=175 deg unattended).

STABILITY - energy tank (the only guaranteed mechanism):
    E += (dissipation by B_L and residual friction) - (power injected into H)
    kappa scales to zero as E hits its floor. Plus: F_hat low-pass + rate
    limit, qdot_L dead-zone, kappa scheduled by sigma_min(J_vL) (wrist-center
    force lines are physically unobservable - detect and degrade).

Measured constants (examples/friction_id.py, 2026-08-20/21):
    breakaway brackets (lo,hi] N.m:  J1 (0.60,0.90]+/(0.40,0.60]-
        J2 (1.40,2.00]+/(2.60,3.20]-   J3 (0.55,0.80]+/(1.10,1.50]-
    kinetic mu_c (min-dir):          [0.31 0.31 0.29 | 0.07 0.06 0.03]
    apparent inertia:                [0.15 1.10 0.31 | 0.030 0.006 0.010]
    L time constants with B_L below: 75/17/33 ms  (<< H admittance bandwidth)

    python examples/hand_guidance.py                     # hand test
    python examples/hand_guidance.py --duration 25 --yes # Test A: no self-drive
    python examples/hand_guidance.py --test-push 3.5 --duration 20 --yes  # Test B
"""
import argparse
import os
import sys

sys.path.insert(0, ".")

_PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
try:
    from piperx_teleop import PiperModel, TorqueSession, require_patched_sdk
    require_patched_sdk()
except (RuntimeError, ImportError):
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

import numpy as np

H, L = [0, 1, 2], [3, 4, 5]

# ---- measured plant ----
MU_C = np.array([0.31, 0.31, 0.29, 0.07, 0.06, 0.03])       # kinetic Coulomb
FS_UB_POS = np.array([0.90, 2.00, 0.80, 0.15, 0.15, 0.14])  # breakaway hi-bound
FS_UB_NEG = np.array([0.60, 3.20, 1.50, 0.15, 0.15, 0.14])
FS_EFF = np.maximum(FS_UB_POS, FS_UB_NEG)                   # for the alpha formula

# ---- Layer 1 design ----
ALPHA_MAX = 0.8
TAU_S_V = (1.0 - ALPHA_MAX) * FS_EFF[H].max()               # uniform floor: 0.64 N.m
ALPHA = np.clip(1.0 - TAU_S_V / FS_EFF, 0.0, ALPHA_MAX)
ALPHA[L] = 0.0                                              # L: no virtual Coulomb
B_L = np.zeros(6)
B_L[L] = [0.40, 0.35, 0.30]                                 # injected damping = sensor
# doc #6: small dither on L ONLY, to erase wrist stiction - without it a 1 N
# push (~0.1 N.m at the wrist) sits below wrist static friction, L never
# slips, and the Layer-2 observer reads silence (measured: direction cos 0.05
# vs the true push). Amplitudes 0.85 x the lower breakaway bound: sub-static,
# cannot self-drive; zero-mean, so the observer's LP/HP averages it out.
# ADAPTIVE dither (closed-loop extension of doc #6, ours): wrist stiction is
# pose-dependent 0.10..0.65, so no fixed amplitude both erases it and never
# self-drives (both failures measured). Regulate each L joint's amplitude to
# a target MICRO-SLIP RATE instead: no slip events -> grow toward the edge;
# events surge (real motion or creep) -> back off. Self-tuning and
# self-stabilizing by construction.
A_MIN, A_MAX = 0.02, np.array([0.0, 0.0, 0.0, 0.14, 0.14, 0.13])
SLIP_V = 0.06            # rad/s: what counts as a slip event
SLIP_TARGET = 1.5        # events/s per joint
A_GAIN = 0.06            # N.m per (event/s error) per second
DITHER_HZ = 31.0
PHI = np.array([0.0, 0.0, 0.0, 1.1, 3.2, 5.3])

# ---- Layer 2 design ----
V_EPS = 0.06          # rad/s: sgn() smoothing for the friction model
V_STUCK = 0.05        # below this a joint counts as stuck
V_DEAD_L = 0.02       # rad/s dead-zone on the observer inputs
F_LP = 0.30           # F_hat low-pass (per 5 ms tick ~ 15 ms time constant)
F_RATE = 40.0         # N/s rate limit on F_hat
DELTA = np.array([0.20, 0.35, 0.25, 0, 0, 0])   # breakaway trigger dead-zone, N.m
SIGMA_LO, SIGMA_HI = 0.010, 0.025               # kappa scheduling on sigma_min(J_vL)
BREAK_RAMP = 0.006                              # s ramp-out after first motion
E0, E_FLOOR = 1.2, 0.0                          # energy tank, joules
LAMBDA = 0.02                                   # damped pseudoinverse

ap = argparse.ArgumentParser(description="two-layer hand guidance")
ap.add_argument("--kappa", type=float, default=1.0, help="force feed-through gain")
ap.add_argument("--k", type=float, default=1.5, help="task-space damping N/(m/s)")
ap.add_argument("--test-push", type=float, default=0.0)
ap.add_argument("--test-pose", type=str, default="0,50,-75,10,15,0")
ap.add_argument("--test-dir", choices=["j1", "observable"], default="j1",
                help="push direction: J1's leverage direction (near the wrist "
                     "blind cone - adversarial) or the most wrist-OBSERVABLE "
                     "direction (top singular vector of J_vL)")
ap.add_argument("--duration", type=float, default=0.0)
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

mdl = PiperModel()
J_EPS = 1e-4
TIP = 0.20


def lowest_z(q):
    T = mdl.fk(q)
    o, z = T[:3, 3], T[:3, 2]
    return float(min(o[2], (o + TIP * z)[2], (o - TIP * z)[2]))


def jacobian(q):
    p0 = mdl.fk(q)[:3, 3]
    J = np.zeros((3, 6))
    for j in range(6):
        qe = q.copy()
        qe[j] += J_EPS
        J[:, j] = (mdl.fk(qe)[:3, 3] - p0) / J_EPS
    return J


class HandGuidance:
    def __init__(self):
        self.v_f = np.zeros(6)
        self.F_hat = np.zeros(3)
        self.tank = E0
        self.state = np.zeros(6)          # 0 stuck, >0: seconds since first motion
        self._dir = None
        self._done = False
        self._q0 = None
        self.log = []
        self.t_prev = None

    def __call__(self, s):
        dt = 0.005 if self.t_prev is None else min(max(s.t - self.t_prev, 1e-3), 0.05)
        self.t_prev = s.t
        self.v_f += 0.5 * (s.qdot - self.v_f)
        v = self.v_f
        J = jacobian(s.q)

        # ---------------- Layer 1 ----------------
        tau = mdl.gravity_torque(s.q)
        f_comp = ALPHA * MU_C * np.tanh(v / V_EPS)          # H friction cancel
        tau += f_comp
        tau -= B_L * v                                       # L damping (sensor)
        # adaptive stiction eraser on L
        if not hasattr(self, "A_d"):
            self.A_d = np.full(6, A_MIN) * (A_MAX > 0)
            self.slip_rate = np.zeros(6)
        slipping = (np.abs(v) > SLIP_V).astype(float)
        self.slip_rate += 2.0 * dt * (slipping / max(dt, 1e-3) * dt * 20 - self.slip_rate)
        err = SLIP_TARGET - self.slip_rate * 20              # events/s estimate
        self.A_d = np.clip(self.A_d + A_GAIN * err * dt * (A_MAX > 0), A_MIN, A_MAX)
        tau += self.A_d * np.sin(2 * np.pi * DITHER_HZ * s.t + PHI) \
               * (1.0 - np.tanh(np.abs(v) / 0.15))
        if a.k > 0:
            tau -= a.k * (J.T @ (J @ v))                     # task-space damping

        # ---------------- Layer 2: observer ----------------
        vL = np.where(np.abs(v[L]) > V_DEAD_L, v[L], 0.0)
        y = B_L[L] * vL                                      # tau_ext on L
        A_ = J[:, L].T                                       # (J_vL)^T, 3x3
        U, S, Vt = np.linalg.svd(A_)
        sigma_min = S.min()
        F_raw = Vt.T @ (S / (S**2 + LAMBDA**2) * (U.T @ y))  # damped pinv
        # HIGH-PASSED observer: a real push is transient at onset (the only
        # moment Layer 2 exists for - breakaway); L-side gravity-model error
        # is quasi-DC creep and must self-cancel. Fast LP minus slow LP.
        # (Without this, ~0.05 N.m of wrist model error reads as a permanent
        # phantom force and walked the arm 25 deg in a no-push soak.)
        if not hasattr(self, "F_slow"):
            self.F_slow = np.zeros(3)
        dF = np.clip(F_raw - self._F_fast if hasattr(self, "_F_fast") else F_raw,
                     -F_RATE * dt, F_RATE * dt)
        self._F_fast = (self._F_fast + F_LP * dF) if hasattr(self, "_F_fast") else F_raw * F_LP
        self.F_slow += 0.003 * (self._F_fast - self.F_slow)   # ~1.7 s
        self.F_hat = self._F_fast - self.F_slow
        kappa = a.kappa * np.clip((sigma_min - SIGMA_LO) / (SIGMA_HI - SIGMA_LO), 0, 1)
        kappa *= np.clip(self.tank / (0.25 * E0), 0.0, 1.0)  # tank scheduling

        tau_ext_hat = J.T @ self.F_hat                       # per-joint external est.
        tau_a = np.zeros(6)
        tau_a[H] = kappa * tau_ext_hat[H]                    # continuous feed-through

        # breakaway events + stick-slip state machine (H only)
        for i in H:
            moving = abs(v[i]) > V_STUCK
            if moving:
                self.state[i] = min(self.state[i] + dt, 1.0) if self.state[i] > 0 else dt
            else:
                self.state[i] = 0.0
            if not moving and abs(tau_ext_hat[i]) > DELTA[i]:
                ub = FS_UB_POS[i] if tau_ext_hat[i] > 0 else FS_UB_NEG[i]
                tau_a[i] += kappa * ub * np.sign(tau_ext_hat[i])
            elif 0 < self.state[i] < BREAK_RAMP:
                ub = FS_UB_POS[i] if v[i] > 0 else FS_UB_NEG[i]
                ramp = 1.0 - self.state[i] / BREAK_RAMP      # -> down to Coulomb
                tau_a[i] += kappa * ramp * max(ub - MU_C[i], 0.0) * np.sign(v[i])

        # ---------------- energy tank, STRICT accounting ----------------
        # No dissipation credit: a self-drive loop dissipates exactly what it
        # injects, so crediting friction losses makes the tank blind to it
        # (measured). Recharge only during verified quiescence.
        inject = float(max(v[H] @ tau_a[H], 0.0))
        quiescent = (np.abs(v[H]) < V_STUCK).all() and np.linalg.norm(self.F_hat) < 0.3
        recharge = 0.15 if quiescent else 0.0
        self.tank = float(np.clip(self.tank + (recharge - inject) * dt, E_FLOOR, E0))
        tau += tau_a

        # ---------------- test harness ----------------
        if a.test_push > 0 and not self._done:
            # push like a FINGER, not a glacier: step onsets (0.3 s rise,
            # 1.5 s hold, 1 s release) at escalating amplitude. The observer
            # is deliberately high-passed (DC model error rejection), so a
            # 10 s ramp self-cancels - measured: HP gain 0.17 on the old ramp,
            # exactly tau_hp/T_ramp. Real pushes are transient; test likewise.
            cyc, ph = divmod(s.t, 2.8)
            level = min((cyc + 1) / 6.0, 1.0) * a.test_push
            if ph < 0.3:
                amp = level * (ph / 0.3)
            elif ph < 1.8:
                amp = level
            else:
                amp = 0.0
            if self._dir is None:
                if a.test_dir == "observable":
                    U_, S_, _ = np.linalg.svd(J[:, L])
                    c = U_[:, 0] * np.sign(U_[0, 0] + 1e-9)
                else:
                    c = J[:, 0].copy()
                c[2] = min(c[2], 0.3)           # cap downward/vertical component
                if c[2] < 0:
                    c[2] = 0.0                  # never push toward the table
                self._dir = c / max(np.linalg.norm(c), 1e-6)
                self._q0 = s.q.copy()
            # the test measures follow thresholds, not workspace tours: stop
            # pushing once the measurement exists or clearance shrinks
            if (np.abs(np.degrees(s.q - self._q0)) > 25).any() or lowest_z(s.q) < 0.08:
                self._done = True
            else:
                tau += J.T @ (amp * self._dir)
            self.log.append((s.t, amp, s.q.copy(), self.F_hat.copy(),
                             (amp * self._dir).copy(), kappa, self.tank))
        return tau


law = HandGuidance()
sess = TorqueSession(law, can=a.can)
print("pose (deg):", np.degrees(sess.q()).round(1))
print("alpha:", ALPHA.round(2), "| tau_s_v floor: %.2f N.m | B_L:" % TAU_S_V, B_L[L])
if not a.yes:
    input(">>> ENTER: two-layer hand guidance live. Push the EE anywhere ")
if a.test_push > 0:
    qt = np.radians([float(x) for x in a.test_pose.split(",")])
    sess._ensure_ready()
    sess._goto_position(qt, secs=3.0)
trip = sess.run(duration=a.duration)
print("position hold restored" + (" (%s)" % trip if trip else ""))

if a.test_push > 0 and law.log:
    Lg = law.log
    t = np.array([r[0] for r in Lg]); amp = np.array([r[1] for r in Lg])
    q = np.array([r[2] for r in Lg]); Fh = np.array([r[3] for r in Lg])
    Ft = np.array([r[4] for r in Lg]); kap = np.array([r[5] for r in Lg])
    tank = np.array([r[6] for r in Lg])
    np.savez("data/hg_test.npz", t=t, amp=amp, q=q, F_hat=Fh, F_true=Ft,
             kappa=kap, tank=tank)
    q0 = q[0]
    print("\nvirtual fingertip (step pushes) along J1's leverage direction:")
    for j in range(6):
        m = np.abs(np.degrees(q[:, j] - q0[j])) > 3.0
        print("  J%d: %s" % (j + 1, "follows at %.2f N" % amp[m.argmax()]
                             if m.any() else "never moved (max %.1f N)" % amp.max()))
    sel = amp > 0.8
    if sel.any():
        err = np.linalg.norm(Fh[sel] - Ft[sel], axis=1)
        w = np.linalg.norm(Fh[sel], axis=1)
        cos = np.sum(Fh[sel] * Ft[sel], 1) / (w * np.linalg.norm(Ft[sel], axis=1) + 1e-9)
        cw = float((cos * w).sum() / (w.sum() + 1e-9))
        print("observer: |F_hat-F| mean %.2f N | cos (magnitude-weighted) %.2f | tank end %.2f J"
              % (err.mean(), cw, tank[-1]))
