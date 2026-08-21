"""Virtual dynamics: precomputed inertia + friction matching, then virtual work.

Layer 1 - model matching, per joint (measured constants from friction_id):
    tau1 = beta_j * M_j * qddot   +  fill_j * tanh(qdot/v_eps)
    - the inertia term pays beta of each joint's REAL apparent inertia during
      acceleration (beta capped ~0.35: qddot comes from double-differencing
      the encoder, and larger cancellation feeds the noise back as buzz)
    - the friction fill is the proven balanced_drag term (alpha-bounded)
    Result: all joints approximate ONE virtual joint (small M*, small f*).

Layer 2 - virtual work on the now-uniform arm (dW = F dx, F = k*I):
    tau2 = -k * J'J * qdot          # isotropic hand drag, host-rendered t_ff

    python examples/virtual_dynamics.py                    # both layers on
    python examples/virtual_dynamics.py --beta 0 --k 0     # = balanced_drag
    python examples/virtual_dynamics.py --k 0              # matching only
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

# measured (examples/friction_id.py); J3 falling-dir and J6 fits were degenerate
M_APP = np.array([0.15, 1.10, 0.31, 0.030, 0.006, 0.010])   # kg m^2
F0 = np.array([0.31, 0.60, 0.29, 0.07, 0.06, 0.00])          # N.m kinetic
F_TARGET, V_EPS = 0.08, 0.08
TAU_INERTIA_CAP = np.array([0.6, 1.2, 0.6, 0.15, 0.05, 0.05])  # noise fuse, N.m

ap = argparse.ArgumentParser(description="virtual inertia+friction, then k*I hand drag")
ap.add_argument("--beta", type=float, default=0.30,
                help="fraction of real inertia to pay on J1-J3 (0..0.4)")
ap.add_argument("--alpha", type=float, default=0.8, help="friction fill fraction")
ap.add_argument("--k", type=float, default=2.0, help="hand drag N/(m/s), 0=off")
ap.add_argument("--duration", type=float, default=0.0)
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

beta = np.clip(a.beta, 0.0, 0.4) * np.array([1, 1, 1, 0, 0, 0])  # shoulder only
FILL = a.alpha * np.maximum(F0 - F_TARGET, 0.0)
mdl = PiperModel()
J_EPS = 1e-4


def jacobian(q):
    p0 = mdl.fk(q)[:3, 3]
    J = np.zeros((3, 6))
    for j in range(6):
        qe = q.copy()
        qe[j] += J_EPS
        J[:, j] = (mdl.fk(qe)[:3, 3] - p0) / J_EPS
    return J


class Law:
    def __init__(self):
        self.v_f = np.zeros(6)      # filtered velocity
        self.a_f = np.zeros(6)      # filtered acceleration
        self.t_prev = None

    def __call__(self, s):
        dt = 0.005 if self.t_prev is None else max(s.t - self.t_prev, 1e-3)
        self.t_prev = s.t
        v_new = 0.5 * self.v_f + 0.5 * s.qdot          # ~30 Hz corner
        acc = (v_new - self.v_f) / dt
        self.v_f = v_new
        self.a_f = 0.85 * self.a_f + 0.15 * acc        # ~10 Hz corner

        tau = mdl.gravity_torque(s.q)
        tau += FILL * np.tanh(self.v_f / V_EPS)                       # friction match
        tau += np.clip(beta * M_APP * self.a_f,
                       -TAU_INERTIA_CAP, TAU_INERTIA_CAP)             # inertia match
        if a.k > 0:
            J = jacobian(s.q)
            tau -= a.k * (J.T @ (J @ self.v_f))                       # virtual work
        return tau


sess = TorqueSession(Law(), can=a.can)
print("pose (deg):", np.degrees(sess.q()).round(1))
print("beta*M (inertia paid):", (beta * M_APP).round(3), " fill:", FILL.round(2),
      " k=%.1f" % a.k)
if not a.yes:
    input(">>> ENTER: matched joints + isotropic hand drag. Accelerate briskly "
          "to feel the inertia term ")
trip = sess.run(duration=a.duration)
print("position hold restored" + (" (%s)" % trip if trip else ""))
