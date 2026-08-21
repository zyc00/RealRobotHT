"""Hand-guidance, the version that works: wrist damping -> projected transfer.

    tau   = G(q)                          # gravity compensation
    tau_L -= B_L * qdot_L                 # J4-J6: damping (feel + sensor)
    F_hat  = pinv(J_L^T) (B_L qdot_L)     # force the damping absorbs,
                                          #   projected to a hand-frame force
    tau_H += ratio * J_H^T F_hat          # re-applied on J1-J3, amplified

Tuning lives in config/hand_guidance.toml (CLI overrides it).

    python examples/damping_transfer.py
    python examples/damping_transfer.py --ratio 3
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
LAMBDA = 0.02                            # damped pinv, near-singularity safety
CONFIG = "config/hand_guidance.toml"

ap = argparse.ArgumentParser(description="wrist damping -> base transfer")
ap.add_argument("--ratio", type=float, default=None,
                help="override [transfer].ratio for all three base joints")
ap.add_argument("--config", default=CONFIG)
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
B_L = np.asarray(cfg.get("wrist", {}).get("damping", [0.40, 0.35, 0.30]), float)
V_DEAD = float(cfg.get("wrist", {}).get("dead_zone", 0.02))
RATIO = np.asarray(cfg.get("transfer", {}).get("ratio", [2.0, 2.0, 2.0]), float) \
        * np.ones(3)
PAIR_J4 = float(cfg.get("transfer", {}).get("pair_j4", 0.0))
if a.ratio is not None:
    RATIO = a.ratio * np.ones(3)

mdl = PiperModel()
J_EPS = 1e-4


def jacobian(q):
    """Position rows (3x6) and unit joint axes (3x6), one fk pass."""
    T0 = mdl.fk(q)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    Jv, W = np.zeros((3, 6)), np.zeros((3, 6))
    for j in range(6):
        qe = q.copy()
        qe[j] += J_EPS
        Te = mdl.fk(qe)
        Jv[:, j] = (Te[:3, 3] - p0) / J_EPS
        dR = Te[:3, :3] @ R0.T
        w = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                            dR[1, 0] - dR[0, 1]]) / J_EPS
        W[:, j] = w / (np.linalg.norm(w) + 1e-9)
    return Jv, W


class Law:
    def __init__(self):
        self.v_f = np.zeros(6)

    def __call__(self, s):
        self.v_f += 0.5 * (s.qdot - self.v_f)
        v = self.v_f
        J, W = jacobian(s.q)

        tau = mdl.gravity_torque(s.q)

        vL = np.where(np.abs(v[L]) > V_DEAD, v[L], 0.0)
        tau[L] -= B_L * v[L]                          # the damping

        # projection: torque absorbed by the damping -> hand force
        A = J[:, L].T                                  # 3x3
        U, S, Vt = np.linalg.svd(A)
        F_hat = Vt.T @ (S / (S**2 + LAMBDA**2) * (U.T @ (B_L * vL)))

        tau[H] += RATIO * (J[:, H].T @ F_hat)          # transfer, amplified

        # paired channel: J4 damping torque -> J2, J3 (front-back pushes);
        # axis dot product carries the correct sign at every pose
        y4 = B_L[0] * vL[0]
        tau[1] += PAIR_J4 * float(W[:, 1] @ W[:, 3]) * y4
        tau[2] += PAIR_J4 * float(W[:, 2] @ W[:, 3]) * y4
        return tau


sess = TorqueSession(Law(), can=a.can)
print("pose (deg):", np.degrees(sess.q()).round(1))
print("ratio:", RATIO, "| pair_j4 %.1f | damping:" % PAIR_J4, B_L, "| dead zone %.3f" % V_DEAD)
if not a.yes:
    input(">>> ENTER: gravity comp + wrist damping -> base transfer. Push the EE ")
trip = sess.run(duration=a.duration)
print("position hold restored" + (" (%s)" % trip if trip else ""))
