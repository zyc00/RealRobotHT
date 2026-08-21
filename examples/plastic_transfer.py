"""Hand-guidance via an elastoplastic wrist spring -> geometric base transfer.

The "low-gear joints compensate high-gear joints" scheme, built from the
three first principles of the design doc:

  P1 single injection point   every element is passive (spring stores,
     damper/plastic slider dissipate, ideal transmission conserves); the ONLY
     energy-producing gain is kappa > 1, and it alone is tank-guarded.
  P2 Lipschitz everywhere     no sgn(), no hard velocity deadbands, no mode
     switches: yield and signal deadband are soft-threshold (shrink)
     operators, continuous through zero.
  P3 frequency layering       wrist spring ~10 Hz >> transfer LPF 2.5 Hz >>
     plastic yield ~0.3 Hz >> recenter (off by default).

Mechanism, per requirement:
  J4-J6 (L, low gear): local spring anchored at q0 + damping. The READING
     Phi_i = K_i (q_i - q0_i) is a static, differentiation-free force sensor
     (unlike damping_transfer.py it sees a push even with the wrist stopped).
     Plastic yield slides q0 so a sustained twist clamps |Phi| at mu_y: the
     wrist turns freely against a constant small resistance, never locks, and
     the base assist can never grow much past stiction.            [req 3]
  J1-J3 (H, high gear): F_hat = pinv(J_vL^T) shrink(Phi, dead) -> LPF ->
     tau_H += sat_taucap(kappa * J_vH^T F_f) - B_H qdot_H, kappa per joint.
     The reading is explained as a hand FORCE at the EE and re-applied on
     the base through the geometry; sign and cross-joint split come from
     the Jacobian at the current pose.                           [req 1, 2]
     (History: an axis-dot moment-matching variant and a direct-pairing +
     dominance-gate variant were tried and rejected by the operator - this
     v1 transfer is the one whose J5->J1 feel they tuned and liked.)
  Not exploding: per-joint cap at 1.3x measured breakaway (a wrong F_hat can
     at worst nudge the base), 2.5 Hz LPF cuts the base-shakes-wrist
     mechanical loop, leaky-bucket tank bounds the long-run average injected
     power, blind-cone truncation near wrist singularity, TorqueSession's
     runaway watchdog last.                                        [req 4]

Deviations from the doc, both deliberate:
  * The wrist spring runs HOST-side in t_ff at 200 Hz, not in the firmware
    MIT kp/kd fields - those units have never passed M0 verification on this
    arm while t_ff is calibrated end-to-end (same flag as hand_admittance).
    K <= 3 N.m/rad with damping is ~10 Hz dynamics; 200 Hz hosting is safe.
  * The gravity base layer is mdl.gravity_torque (piper_x URDF, holds every
    joint to ~0.3 deg pure) - record_drag.py's firmware drag mode (0xFA)
    cannot accept injected torques, so it cannot host this controller.

Tuning lives in config/plastic_transfer.toml (CLI overrides it).

    python examples/plastic_transfer.py
    python examples/plastic_transfer.py --kappa 1,4,2.5
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
import tomllib

H, L = [0, 1, 2], [3, 4, 5]
FS_UB_H = np.array([0.90, 3.20, 1.50])       # measured J1-J3 breakaway upper bounds
CONFIG = "config/plastic_transfer.toml"

ap = argparse.ArgumentParser(description="elastoplastic wrist -> base transfer")
ap.add_argument("--kappa", type=str, default=None,
                help="override [transfer].kappa: one value or 'k1,k2,k3' per base joint")
ap.add_argument("--config", default=CONFIG)
ap.add_argument("--log", default="", help="save a diagnostic npz here on exit")
ap.add_argument("--duration", type=float, default=0.0)
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

cfg = {}
if os.path.exists(a.config):
    with open(a.config, "rb") as f:
        cfg = tomllib.load(f)
    print("config:", a.config)
else:
    print("config: %s not found - built-in defaults" % a.config)


def _get(sec, key, default):
    return np.asarray(cfg.get(sec, {}).get(key, default), float)


K_L = _get("wrist", "stiffness", [3.0, 3.0, 2.0])       # N.m/rad, the sensor
B_LW = _get("wrist", "damping", [0.10, 0.15, 0.10])     # N.m/(rad/s)
MU_Y = _get("wrist", "yield", [0.06, 0.06, 0.04])       # N.m, sustained-twist feel
PHI_DEAD = _get("wrist", "dead", [0.06, 0.05, 0.04])    # N.m, swallows gravity-model bias
PHI_MAX = 3.0 * MU_Y                                     # elastic hard clip (yank bound)
ETA = float(_get("plastic", "eta", 0.6))                 # s, yield relaxation
T_R = float(_get("plastic", "recenter_s", 0.0))          # s, q0 recenter; 0 = off
KAPPA = _get("transfer", "kappa", [1.0, 4.0, 2.5]) * np.ones(3)   # per J1/J2/J3
if a.kappa is not None:
    KAPPA = np.array([float(x) for x in a.kappa.split(",")]) * np.ones(3)
J5_ONLY_J1 = bool(cfg.get("transfer", {}).get("j5_only_j1", True))
FC = float(_get("transfer", "lpf_hz", 2.5))
TAU_CAP = float(_get("transfer", "cap_ratio", 1.3)) * FS_UB_H
B_H = _get("transfer", "base_damping", [0.3, 0.8, 0.5])  # grounding damping
E_TANK0 = float(_get("tank", "energy", 12.0))

LAMBDA_FRAC = 0.08
SIGMA_BLIND = 0.012
J_EPS = 1e-4

# q0 must never plastically walk into a limit. piper_x wrist limits from the
# URDF: J4/J5 +/-89 deg, J6 +/-120 deg - the package JOINT_LIMITS still
# carries piper values (J5 +/-70!) and clamping q0 20 deg away from a parked
# J5=85 deg manufactured a permanent phantom reading that drove J1 at rest.
QLIM_L = np.radians([[-89.0, 89.0], [-89.0, 89.0], [-120.0, 120.0]])
Q0_LO, Q0_HI = QLIM_L[:, 0] + np.radians(4), QLIM_L[:, 1] - np.radians(4)

mdl = PiperModel()


def shrink(x, mu):
    """Soft threshold: 0 inside +/-mu, |x|-mu beyond. Lipschitz, no sgn jump."""
    return np.sign(x) * np.maximum(np.abs(x) - mu, 0.0)


def jacobian(q):
    """Linear-velocity rows only (3x6): forces at the EE, no moment channel."""
    p0 = mdl.fk(q)[:3, 3]
    J = np.zeros((3, 6))
    for j in range(6):
        qe = q.copy()
        qe[j] += J_EPS
        J[:, j] = (mdl.fk(qe)[:3, 3] - p0) / J_EPS
    return J


class PlasticTransfer:
    def __init__(self):
        self.v_f = np.zeros(6)
        self.q0 = None                   # wrist spring anchor (the plastic state)
        self.q_mid = None
        self.F_f = np.zeros(3)           # filtered hand-force estimate (full)
        self.F_f23 = np.zeros(3)         # same, J5 channel zeroed (feeds J2/J3)
        self.tank = E_TANK0
        self.kappa_eff = KAPPA.copy()
        self.t_prev = None
        self.n = 0
        self.log = []

    def __call__(self, s):
        dt = 0.005 if self.t_prev is None else min(max(s.t - self.t_prev, 1e-3), 0.05)
        self.t_prev = s.t
        self.v_f += 0.5 * (s.qdot - self.v_f)
        v, q = self.v_f, s.q
        if self.q0 is None:
            self.q0 = q[L].copy()        # Phi starts at exactly 0: no start jump
            self.q_mid = q[L].copy()

        tau = mdl.gravity_torque(q)

        # ---- wrist spring + damping (fast layer; host-side t_ff stand-in) ----
        e = np.clip(q[L] - self.q0, -PHI_MAX / K_L, PHI_MAX / K_L)
        self.q0 = q[L] - e               # write-back: a yank saturates, not winds
        Phi = K_L * e                    # the reading: torque the hand transmits
        tau[L] += -Phi - B_LW * v[L]

        # ---- plastic yield (slow layer): steady reading clamps at mu_y ----
        self.q0 += (shrink(Phi, MU_Y) / (ETA * K_L)) * dt
        if T_R > 0:
            self.q0 -= (self.q0 - self.q_mid) / T_R * dt
        # widen the box to include the ACTUAL position: the clip only stops
        # q0 from walking past a limit - if the wrist is parked beyond the
        # margin, q0 must still be allowed to sit at q or the clip itself
        # fabricates a permanent reading (measured: J5=85 vs stale 70-deg
        # limit -> -0.93 N.m phantom J1 drive at rest)
        self.q0 = np.clip(self.q0, np.minimum(Q0_LO, q[L]), np.maximum(Q0_HI, q[L]))

        # ---- reading -> hand force (deadband kills gravity-model bias) ----
        J = jacobian(q)
        A = J[:, L].T                                    # 3x3: Phi = A F
        U, S, Vt = np.linalg.svd(A)
        # A is structurally rank-2 at most poses (J6's linear lever ~ 0), so
        # reconstruct F in the OBSERVABLE subspace and truncate the rest - a
        # full-rank blind gate would zero the signal permanently
        keep = S > SIGMA_BLIND
        blind = not keep.any()
        lam = LAMBDA_FRAC * (S.max() + 1e-9)
        gain = np.where(keep, S / (S**2 + lam**2), 0.0)
        Phi_sig = shrink(Phi, PHI_DEAD)
        F_hat = Vt.T @ (gain * (U.T @ Phi_sig))
        self.F_f += (1.0 - np.exp(-2 * np.pi * FC * dt)) * (F_hat - self.F_f)
        # the reconstruction is LINEAR in the channels, so J2/J3 can be fed
        # the force with the J5 channel zeroed: rotating J5 then cannot
        # reach J2/J3 at all, while J1 keeps the full v1 estimate/feel and
        # the J4 -> J2/J3 path is untouched
        if J5_ONLY_J1:
            F_no5 = Vt.T @ (gain * (U.T @ (Phi_sig * np.array([1.0, 0.0, 1.0]))))
            self.F_f23 += (1.0 - np.exp(-2 * np.pi * FC * dt)) * (F_no5 - self.F_f23)
        else:
            self.F_f23 = self.F_f

        # ---- transfer to the base (mid layer): capped, grounded ----
        m1 = J[:, H].T @ self.F_f
        m23 = J[:, H].T @ self.F_f23
        tau_tr = np.clip(self.kappa_eff * np.array([m1[0], m23[1], m23[2]]),
                         -TAU_CAP, TAU_CAP)
        tau[H] += tau_tr - B_H * v[H]

        # ---- energy tank: LEAKY BUCKET, strict drain on the excess ----
        # Base recharge is ALWAYS on, bounding the long-run AVERAGE injected
        # power at 0.8 W (a runaway needs more, and the plastic yield denies
        # it: any sustained reading decays to yield level in ~1 s, whose
        # transfer is below base friction). One-shot traps found the hard
        # way: drain scales with (1-1/kappa) so kappa=1 never drains; and
        # never gate recharge on "hands off" via Phi_sig < 0.01 - grip
        # pressure alone keeps the reading above that.
        inject = float(np.sum(np.maximum(v[H] * tau_tr, 0.0)
                              * (1.0 - 1.0 / np.maximum(np.abs(self.kappa_eff), 1.0))))
        hands_off = np.abs(Phi_sig).max() < 0.01
        self.tank = float(np.clip(
            self.tank + ((0.8 + (0.5 if hands_off else 0.0)) - inject) * dt,
            0.0, E_TANK0))
        beta = float(np.clip(self.tank / (0.3 * E_TANK0), 0.0, 1.0))
        self.kappa_eff = 1.0 + beta * (KAPPA - 1.0)      # drained tank -> passive

        self.n += 1
        if a.log and self.n % 4 == 0:
            self.log.append((s.t, Phi.copy(), self.F_f.copy(), tau_tr.copy(),
                             self.tank, float(blind)))
        if self.n % 400 == 0:
            print("  t=%5.1f  |F|=%4.1f N  Phi=%s  tau_H=%s  tank=%.2f%s"
                  % (s.t, np.linalg.norm(self.F_f), Phi.round(2), tau_tr.round(2),
                     self.tank, "  BLIND" if blind else ""))
        return tau


law = PlasticTransfer()
sess = TorqueSession(law, can=a.can)
print("pose (deg):", np.degrees(sess.q()).round(1))
print("K:", K_L, "| mu_y:", MU_Y, "| kappa:", KAPPA.round(1), "| tau_cap:", TAU_CAP.round(2))
print("elastic band at yield (deg):", np.degrees(MU_Y / K_L).round(1))
# a SUSTAINED push is clamped to mu_y by the plastic, then dead is subtracted;
# transient pushes carry up to 3*mu_y - dead for ~eta seconds
steady = shrink(MU_Y, PHI_DEAD)
print("steady signal after yield+dead (N.m):", steady.round(3),
      "| transient up to:", shrink(PHI_MAX, PHI_DEAD).round(2))
if not a.yes:
    input(">>> ENTER: elastoplastic wrist -> base transfer live. Push the EE ")
trip = sess.run(duration=a.duration)
print("position hold restored" + (" (%s)" % trip if trip else ""))

if a.log and law.log:
    t = np.array([r[0] for r in law.log])
    np.savez(a.log, t=t,
             Phi=np.array([r[1] for r in law.log]),
             F_f=np.array([r[2] for r in law.log]),
             tau_tr=np.array([r[3] for r in law.log]),
             tank=np.array([r[4] for r in law.log]),
             blind=np.array([r[5] for r in law.log]))
    print("log: %d samples -> %s" % (len(t), a.log))
