"""Hand-guidance admittance per the M-doc: wrist sensing + base ramp servo.

Architecture (doc sections in brackets):
  J4-J6 (L): damping b_v = the force sensor; gravity ff; kinetic friction
             partially compensated so the residual is small        [3.1, 4.1]
  J1-J3 (H): POSITION RAMP SERVO - the admittance velocity integrates into a
             reference p_d; when a joint sticks, e = p_d - q grows and kp*e
             climbs past stiction BY ITSELF. No mu_s calibration, no
             breakaway feed-forward, no stick-slip state machine.     [4.3]
             Anti-windup: e clamped to +/-e_max and p_d written back, so
             tau_cap = kp * e_max is a design constant, not an outcome.
  Outer loop: xdot_d = F_hat / D_x;
             qdot_d,H = pinv(J_H)(xdot_d - J_L qdot_L) + slow wrist recenter
             (recenter frozen while a push is active)                [4.2]
  Energy tank on servo injection, beta-shrink                        [4.4]
  Safety ladder: clamp-saturation, blind cone, speed, tank        [5: S1-S5]

Deviation from the doc (flagged): the H servo runs HOST-side in t_ff with an
explicit torque clamp, not via the firmware kp field - the MIT kp/kd units
have never passed the doc's own M0 verification on this arm, while t_ff is
calibrated end-to-end. Same law, trusted units. Migrate after M0.

    python examples/hand_admittance.py                       # hand test
    python examples/hand_admittance.py --duration 30 --yes   # Test A
    python examples/hand_admittance.py --test-push 3 --duration 20 --yes  # B
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

# ---- measured constants ----
MU_C_L = np.array([0.07, 0.06, 0.03])           # wrist kinetic friction
FS_UB_H = np.array([0.90, 3.20, 1.50])          # H breakaway upper bounds (coarse!)
B_V = np.array([0.40, 0.35, 0.30])              # wrist damping = sensor
V_NOISE = 0.004                                  # rad/s, measured quantization floor

# ---- design constants (doc 4.3: tau_cap FIRST, then kp, e_max follows) ----
TAU_CAP = 1.2 * FS_UB_H                          # [1.08, 3.84, 1.80]
KP_H = np.array([8.0, 20.0, 12.0])               # N.m/rad, host-side
E_MAX = TAU_CAP / KP_H
M_H = np.array([0.15, 1.10, 0.31])
KD_H = 0.5 * 2 * np.sqrt(KP_H * M_H)             # ~critical/2: [0.55, 2.35, 0.96]
QD_MAX = np.array([0.6, 0.5, 0.6])               # rad/s outer-loop limit
D_X = 40.0                                       # N/(m/s): THE feel knob
V_DEAD = 6.0 * V_NOISE                           # per-channel observer dead-zone
F_LPF = 0.25                                     # ~15 ms at 200 Hz
LAMBDA_FRAC = 0.08
SIGMA_BLIND = 0.012
K_R = 0.4                                        # wrist recenter, 1/s (slow)
F_ACTIVE = 0.35                                  # N: push considered active
E_TANK0 = 3.0
# Wrist kinetic comp DISABLED: a tanh(v/0.05) Coulomb term at the velocity
# noise floor is a +/-0.056 N.m random pump - it broke J4 loose repeatedly
# (81 deg walk in a no-push soak) and its creep became phantom F_hat. The
# sensor channels must carry NOTHING signal-shaped that is not external.
ALPHA_L = 0.0

ap = argparse.ArgumentParser(description="wrist-sensed base-servo admittance")
ap.add_argument("--dx", type=float, default=D_X)
ap.add_argument("--test-push", type=float, default=0.0)
ap.add_argument("--test-dir", choices=["j1", "observable"], default="j1")
ap.add_argument("--test-pose", type=str, default="0,50,-75,10,15,0")
ap.add_argument("--duration", type=float, default=0.0)
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

mdl = PiperModel()
J_EPS = 1e-4
TIP = 0.20


def jacobian(q):
    p0 = mdl.fk(q)[:3, 3]
    J = np.zeros((3, 6))
    for j in range(6):
        qe = q.copy()
        qe[j] += J_EPS
        J[:, j] = (mdl.fk(qe)[:3, 3] - p0) / J_EPS
    return J


def lowest_z(q):
    T = mdl.fk(q)
    o, z = T[:3, 3], T[:3, 2]
    return float(min(o[2], (o + TIP * z)[2], (o - TIP * z)[2]))


class HandAdmittance:
    def __init__(self):
        self.v_f = np.zeros(6)
        self.F_hat = np.zeros(3)
        self.p_d = None                  # H ramp reference
        self.qL0 = None                  # wrist recenter anchor
        self.tank = E_TANK0
        self.beta = 1.0
        self.sat_time = 0.0
        self.t_prev = None
        self._dir = None
        self._done = False
        self._q0 = None
        self.log = []

    def __call__(self, s):
        dt = 0.005 if self.t_prev is None else min(max(s.t - self.t_prev, 1e-3), 0.05)
        self.t_prev = s.t
        self.v_f += 0.5 * (s.qdot - self.v_f)
        v = self.v_f
        q = s.q
        if self.p_d is None:
            self.p_d = q[H].copy()
            self.qL0 = q[L].copy()
        J = jacobian(q)

        # ---------------- wrist: sensor + comfort ----------------
        tau = mdl.gravity_torque(q)
        tau[L] += ALPHA_L * MU_C_L * np.tanh(v[L] / 0.05)   # kinetic comp
        tau[L] -= B_V * v[L]                                 # damping = sensing

        # ---------------- force observer [4.1] ----------------
        vL = np.where(np.abs(v[L]) > V_DEAD, v[L], 0.0)
        y = B_V * vL
        A_ = J[:, L].T
        Wd = np.diag(1.0 / np.maximum(B_V * V_DEAD, 1e-4) ** 2)
        U, S, Vt = np.linalg.svd(np.sqrt(Wd) @ A_)
        lam = LAMBDA_FRAC * (S.max() + 1e-9)
        F_raw = Vt.T @ (S / (S**2 + lam**2) * (U.T @ (np.sqrt(Wd) @ y)))
        sigma_min = np.linalg.svd(A_, compute_uv=False).min()
        blind = sigma_min < SIGMA_BLIND
        if blind:
            F_raw = np.zeros(3)
        self.F_hat += F_LPF * (F_raw - self.F_hat)
        active = np.linalg.norm(self.F_hat) > F_ACTIVE

        # ---------------- admittance outer loop [4.2] ----------------
        xdot_d = self.beta * self.F_hat / a.dx
        resid = xdot_d - J[:, L] @ v[L]
        JH = J[:, H]
        U2, S2, Vt2 = np.linalg.svd(JH, full_matrices=False)
        lam2 = 0.08 * (S2.max() + 1e-9)
        qd_H = Vt2.T @ (S2 / (S2**2 + lam2**2) * (U2.T @ resid))
        if not active:                              # slow wrist recenter
            qd_H += Vt2.T @ (S2 / (S2**2 + lam2**2)
                             * (U2.T @ (J[:, L] @ (K_R * (self.qL0 - q[L])))))
        qd_H = np.clip(qd_H, -QD_MAX, QD_MAX)

        # ---------------- H ramp servo [4.3] ----------------
        self.p_d += qd_H * dt
        e = np.clip(self.p_d - q[H], -E_MAX, E_MAX)
        self.p_d = q[H] + e                          # anti-windup write-back
        tau_servo = KP_H * e - KD_H * v[H]
        tau_servo = np.clip(tau_servo, -TAU_CAP, TAU_CAP)
        tau[H] += tau_servo

        # ---------------- energy tank [4.4] ----------------
        # STRICT accounting (deviation from doc 4.4, twice-proven necessary
        # on this arm): no dissipation credit - a self-drive loop dissipates
        # exactly what it injects and a credited tank never sees it. Recharge
        # only during verified quiescence.
        inject = float(max(v[H] @ tau_servo, 0.0))
        quiescent = np.abs(v).max() < 0.02 and not active
        self.tank = float(np.clip(self.tank + ((0.25 if quiescent else 0.0) - inject) * dt,
                                  0.0, E_TANK0))
        self.beta = float(np.clip(self.tank / (0.3 * E_TANK0), 0.0, 1.0))

        # ---------------- safety ladder (S1, S3 handled via beta) -----------
        saturated = (np.abs(e) >= E_MAX * 0.999).any() and np.abs(v[H]).max() < 0.02
        self.sat_time = self.sat_time + dt if saturated else 0.0
        if self.sat_time > 0.5:                      # S1: pushing a wall
            self.p_d = q[H] + 0.5 * e                # bleed the clamp
        if blind:
            self.beta = 0.0

        # ---------------- test harness ----------------
        if a.test_push > 0 and not self._done:
            cyc, ph = divmod(s.t, 2.8)
            level = min((cyc + 1) / 6.0, 1.0) * a.test_push
            amp = level * (ph / 0.3) if ph < 0.3 else (level if ph < 1.8 else 0.0)
            if self._dir is None:
                if a.test_dir == "observable":
                    U_, _, _ = np.linalg.svd(J[:, L])
                    c = U_[:, 0] * np.sign(U_[0, 0] + 1e-9)
                else:
                    c = J[:, 0].copy()
                c[2] = 0.0 if c[2] < 0 else min(c[2], 0.3)
                self._dir = c / max(np.linalg.norm(c), 1e-6)
                self._q0 = q.copy()
            if (np.abs(np.degrees(q - self._q0)) > 25).any() or lowest_z(q) < 0.08:
                self._done = True
            else:
                tau += J.T @ (amp * self._dir)
            self.log.append((s.t, amp, q.copy(), self.F_hat.copy(),
                             (amp * (self._dir if self._dir is not None else np.zeros(3))).copy(),
                             e.copy(), self.tank))
        return tau


law = HandAdmittance()
sess = TorqueSession(law, can=a.can)
print("pose (deg):", np.degrees(sess.q()).round(1))
print("tau_cap:", TAU_CAP.round(2), "| e_max (deg):", np.degrees(E_MAX).round(1),
      "| D_x %.0f N/(m/s)" % a.dx)
if not a.yes:
    input(">>> ENTER: wrist-sensed admittance live. Push the EE ")
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
    Ft = np.array([r[4] for r in Lg]); e = np.array([r[5] for r in Lg])
    tank = np.array([r[6] for r in Lg])
    np.savez("data/ha_test.npz", t=t, amp=amp, q=q, F_hat=Fh, F_true=Ft, e=e, tank=tank)
    q0 = q[0]
    print("\nstep pushes, direction=%s:" % a.test_dir)
    for j in range(6):
        m = np.abs(np.degrees(q[:, j] - q0[j])) > 3.0
        print("  J%d: %s" % (j + 1, "follows at %.2f N" % amp[m.argmax()]
                             if m.any() else "never moved (max %.1f N)" % amp.max()))
    sel = amp > 0.5
    if sel.any():
        w = np.linalg.norm(Fh[sel], axis=1)
        cos = np.sum(Fh[sel] * Ft[sel], 1) / (w * np.linalg.norm(Ft[sel], axis=1) + 1e-9)
        cw = float((cos * w).sum() / (w.sum() + 1e-9))
        print("observer weighted cos %.2f | peak servo |e| %s deg | tank end %.2f J"
              % (cw, np.degrees(np.abs(e).max(0)).round(1), tank[-1]))
