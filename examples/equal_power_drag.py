"""Equal power on each joint, by virtual work. The user's derivation, verbatim:

    dW = F' dx,  with the hand drag F = k * I (isotropic)
    dx = J dq,   tau' dq = F' dx  for all dq        (principle of virtual work)
    =>  tau_drag = J' F = -k * J'J * qdot

No further design: the Jacobian itself distributes the hand's work across the
joints - joint j receives exactly the power its motion contributes at the
hand. The hand feels the same drag k in every direction, at every pose.

Control frequency (the real constraint, as the user noted): friction-like
rendering wants a faster loop than 200 Hz CAN allows. Split the term:
  * diag(J'J) -> the firmware's per-joint kd field, rendered AT MOTOR RATE
  * off-diagonal coupling -> host t_ff at 200 Hz (benign for viscous terms)
Optional --coulomb renders F = k * xdot/|xdot| (constant-magnitude drag,
closer to "friction" but rate-sensitive; expect some buzz at 200 Hz).

Native joint friction still exists underneath; --alpha (default 0.8) applies
the measured balanced_drag under-fill so the rendered k*I dominates the feel.
Set --alpha 0 to feel the pure rendered term on top of raw friction.

    python examples/equal_power_drag.py --k 8       # N per m/s of hand speed
    python examples/equal_power_drag.py --k 15 --coulomb
"""
import argparse
import os
import sys

sys.path.insert(0, ".")

_PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
try:
    from piperx_teleop import MitCommand, PiperModel, TorqueSession, require_patched_sdk
    require_patched_sdk()
except (RuntimeError, ImportError):
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

import numpy as np

F0 = np.array([0.31, 0.60, 0.29, 0.07, 0.06, 0.00])   # measured kinetic friction
V_EPS = 0.08
KD_MAX = 4.5           # firmware kd field ceiling

ap = argparse.ArgumentParser(description="isotropic hand drag via virtual work")
ap.add_argument("--k", type=float, default=3.0,
                help="hand drag: N per m/s of hand speed (viscous mode)")
ap.add_argument("--coulomb", action="store_true",
                help="render constant-magnitude drag k [N] instead (rate-sensitive)")
ap.add_argument("--alpha", type=float, default=0.8,
                help="under-fill of native joint friction (0 = off)")
ap.add_argument("--firmware-kd", action="store_true",
                help="EXPERIMENTAL: render diag(J'J) via the firmware kd field. "
                     "That field's units are NOT calibrated N.m.s/rad (measured: "
                     "kd=1.0 is far heavier than 1 N.m.s/rad) - the default "
                     "renders everything host-side in t_ff, whose units ARE "
                     "verified. Viscous terms tolerate 200 Hz fine.")
ap.add_argument("--duration", type=float, default=0.0)
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

mdl = PiperModel()
FILL = a.alpha * np.maximum(F0 - 0.08, 0.0)
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
        self.qdot_f = np.zeros(6)      # light smoothing on the fd velocity

    def __call__(self, s):
        self.qdot_f = 0.6 * self.qdot_f + 0.4 * s.qdot
        qd = self.qdot_f
        J = jacobian(s.q)
        JJ = J.T @ J

        tau = mdl.gravity_torque(s.q)
        tau += FILL * np.tanh(qd / V_EPS)              # native-friction under-fill

        if a.coulomb:
            xdot = J @ qd
            speed = np.linalg.norm(xdot)
            if speed > 1e-3:
                tau -= a.k * (J.T @ (xdot / max(speed, 0.03)))
            return tau

        # viscous: tau_drag = -k J'J qdot, rendered host-side in t_ff where
        # the units are verified N.m (the firmware kd field is NOT label
        # units - it made J1-J3 nearly immovable when trusted).
        if not a.firmware_kd:
            return tau - a.k * (JJ @ qd)
        kd_fw = np.minimum(a.k * np.diag(JJ), KD_MAX)
        off = JJ - np.diag(np.diag(JJ))
        tau -= a.k * (off @ qd)
        tau -= np.maximum(a.k * np.diag(JJ) - kd_fw, 0.0) * qd
        return MitCommand(t_ff=tau, kd=kd_fw)


sess = TorqueSession(Law(), can=a.can)
q = sess.q()
JJ = (lambda J: J.T @ J)(jacobian(q))
print("pose (deg):", np.degrees(q).round(1))
print("k=%.1f %s | diag(J'J)=%s -> firmware kd %s" %
      (a.k, "coulomb" if a.coulomb else "viscous",
       np.diag(JJ).round(3), np.minimum(a.k * np.diag(JJ), KD_MAX).round(2)))
if not a.yes:
    input(">>> ENTER to go compliant - hand drag should be uniform in all directions ")
trip = sess.run(duration=a.duration)
print("position hold restored" + (" (%s)" % trip if trip else ""))
