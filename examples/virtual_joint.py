"""The general algorithm: six uniform virtual joints + virtual work.

Reference model (design): every joint presents the SAME virtual dynamics
    M* qddot + b* qdot + fk* sgn(qdot) = tau_ext,   static friction fs*
and the hand feels F = -k * I * xdot (virtual work, dW = F dx).

Law = target minus measured reality, term by term (all constants measured by
examples/friction_id.py; nothing tuned to any specific push or contact point):

    tau = G(q)                                       gravity
        + beta_lim (M_j - M*) qddot_hat              inertia matching
        + (f0_j - fk*) tanh(qdot/ve) + (b_j - b*) qdot   kinetic matching
        + A_j sin(w t + phi_j) (1 - tanh(|qdot|/vs))     stiction matching
        + J' (-k J qdot)                             virtual work

Stiction matching: A_j = fs_j - fs* is the measured EXCESS static friction.
Zero-mean dither of that amplitude keeps each joint at the edge of slipping:
alone it never breaks away (peak < fs_j), but any external share >= fs* tips
the peaks over. Effective stiction becomes fs* on EVERY joint - a fingertip's
J'-share then rotates J1 like anything else. The cost is honest: a faint
28 Hz buzz at rest, scaled by --dither.

    python examples/virtual_joint.py                     # full algorithm
    python examples/virtual_joint.py --dither 0          # no stiction term
    python examples/virtual_joint.py --k 0 --beta 0 --dither 0   # ~drag mode
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

# ---- measured plant (friction_id) ----
# RULE: every value used for cancellation is the per-direction MINIMUM of the
# measurement. A cancellation built on an average over-compensates in the
# weaker direction and becomes an energy source ("hard to stop" - felt).
M_APP = np.array([0.15, 1.10, 0.31, 0.030, 0.006, 0.010])   # kg m^2
F0    = np.array([0.31, 0.31, 0.29, 0.07, 0.06, 0.03])      # kinetic, min-dir
FS    = np.array([0.40, 1.40, 0.55, 0.10, 0.10, 0.10])      # breakaway, min-dir
# Per-direction breakaway BRACKETS (largest-no-move, smallest-moved) from the
# ID data. The directed assist uses the bracket MIDPOINT per direction: a cap
# built on the lower bound is guaranteed below true static friction and
# therefore guaranteed to never finish the job (measured: J1+ needs 0.6-0.9,
# lower-bound cap gave 0.36 - the fingertip was paying the difference).
FS_POS = np.array([0.75, 1.70, 0.675, 0.40, 0.45, 0.12])     # mid of (+) bracket
FS_NEG = np.array([0.50, 2.90, 1.30,  0.40, 0.45, 0.12])     # mid of (-) bracket
# wrist directed caps use the STICKY-pose end of their range: J4/J5 static
# friction is position-dependent up to ~0.65 (measured), and the directed
# term is motion-gated, so the at-rest self-drive bound does not apply to it.
# viscous matching is DISABLED: the b fits disagreed in SIGN between
# directions (J2: +0.60 vs -0.84) - cancelling an unmeasured quantity is how
# a compensator becomes a motor. Re-enable only after a clean re-fit.

# ---- the virtual joint (design targets) ----
M_STAR, FK_STAR, FS_STAR = 0.02, 0.06, 0.08
# The ADD half of the design (the part that makes force TRANSFER work):
# joints BELOW the virtual targets get friction/damping RAISED to them.
# Without this the wrist yields for free after breakaway, the contact force
# collapses, and nothing distal of your fingertip ever reaches the shoulder's
# threshold - "small damping on J4/5 compensates large stiction on J1/2".
FK_ADD_T = 0.10        # virtual kinetic floor, N.m (wrist rises TO this)
B_ADD_T = 0.30         # virtual viscous floor, N.m s (pure addition: safe)

V_EPS, V_STIC = 0.08, 0.12
DITHER_HZ = 28.0
PHI = np.array([0.0, 2.1, 4.2, 1.0, 3.1, 5.2])       # decorrelate joints
TAU_INERTIA_CAP = np.array([0.6, 1.2, 0.6, 0.15, 0.05, 0.05])

ap = argparse.ArgumentParser(description="uniform virtual joints + virtual work")
ap.add_argument("--k", type=float, default=2.0, help="hand drag, N/(m/s)")
ap.add_argument("--beta", type=float, default=0.30, help="inertia matching cap (<=0.4)")
ap.add_argument("--dither", type=float, default=0.85,
                help="fraction of the min-direction stiction excess to dither away")
ap.add_argument("--test-pose", type=str, default="0,50,-75,20,10,0",
                help="pose (deg, comma list) to run the virtual-fingertip test from")
ap.add_argument("--test-push", type=float, default=0.0,
                help="VIRTUAL FINGERTIP: ramp a simulated EE force 0->this many "
                     "newtons along J1's leverage direction; logs which joints "
                     "follow at what force. The arm WILL move by itself.")
ap.add_argument("--duration", type=float, default=0.0)
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

beta = np.clip(a.beta, 0.0, 0.4)
A_DITHER = np.clip(a.dither, 0.0, 0.85) * np.maximum(FS - FS_STAR, 0.0)
FILL_K = np.maximum(F0 - FK_STAR, 0.0)          # cancel down (min-direction)
ADD_K = np.maximum(FK_ADD_T - F0, 0.0)          # raise up (pure dissipation)
ADD_B = np.where(F0 < FK_ADD_T, B_ADD_T, 0.0)   # damping where friction is low
DM = np.maximum(M_APP - M_STAR, 0.0) * beta
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
        self.v_f = np.zeros(6)
        self.a_f = np.zeros(6)
        self.t_prev = None
        self._J = np.zeros((3, 6))
        self._dir = None
        self.log = []

    def __call__(self, s):
        dt = 0.005 if self.t_prev is None else max(s.t - self.t_prev, 1e-3)
        self.t_prev = s.t
        v_new = 0.5 * self.v_f + 0.5 * s.qdot
        self.a_f = 0.85 * self.a_f + 0.15 * (v_new - self.v_f) / dt
        self.v_f = v_new
        v = self.v_f

        tau = mdl.gravity_torque(s.q)
        tau += np.clip(DM * self.a_f, -TAU_INERTIA_CAP, TAU_INERTIA_CAP)
        tau += (FILL_K - ADD_K) * np.tanh(v / V_EPS)   # down AND up
        tau -= ADD_B * v                               # sustained transfer force
        # stiction matching has TWO regimes. At rest the direction of the
        # coming push is unknown, so the term must stay sub-breakaway
        # (undirected dither). Once ANY motion exists, xdot = J qdot reveals
        # the direction, and each stuck joint's signed share (J'J qdot)_j
        # licenses near-full directed cancellation - it points along observed
        # external motion, stays below the joint's own static friction, and
        # self-arrests when coordination is reached.
        J = jacobian(s.q)
        xdot = J @ v
        # two-scale gate: cm/s motion opens it instantly; mm-scale COHERENT
        # creep (the micro-ratchet a sub-breakaway push produces through the
        # dither) opens it within ~a second. Encoders cannot see a push that
        # produces zero motion - creep is the smallest observable signature.
        self._creep = 0.995 * self._creep + 0.005 * xdot if hasattr(self, "_creep") else xdot * 0.0
        g = max(np.tanh(np.linalg.norm(xdot) / 0.006),
                np.tanh(np.linalg.norm(self._creep) / 0.002))
        share = J.T @ xdot
        undirected = A_DITHER * np.sin(2 * np.pi * DITHER_HZ * s.t + PHI)
        sh = np.tanh(share / 0.015)
        directed = np.where(sh > 0, FS_POS, FS_NEG) * sh
        tau += ((1.0 - g) * undirected + g * directed) \
               * (1.0 - np.tanh(np.abs(v) / V_STIC))
        self._J = J
        if a.k > 0:
            tau -= a.k * (self._J.T @ (self._J @ v))
        if a.test_push > 0:
            amp = a.test_push * min(s.t / 10.0, 1.0)
            if self._dir is None:
                c = J[:, 0]
                self._dir = c / max(np.linalg.norm(c), 1e-6)
            tau += J.T @ (amp * self._dir)
            self.log.append((s.t, amp, s.q.copy()))
        return tau


sess = TorqueSession(Law(), can=a.can)
print("pose (deg):", np.degrees(sess.q()).round(1))
print("virtual joint: M*=%.3f fk*=%.2f fs*=%.2f | k=%.1f" %
      (M_STAR, FK_STAR, FS_STAR, a.k))
print("dither (N.m):", A_DITHER.round(2), "| fill down:", FILL_K.round(2),
      "| raise up: fk", ADD_K.round(2), "b", ADD_B.round(2))
if not a.yes:
    input(">>> ENTER. Then try the fingertip test on the EE - J1 should follow ")
law = sess.law
if a.test_push > 0:
    # extended pose: honest lever arms for the test
    qt = np.radians([float(x) for x in a.test_pose.split(",")])
    sess._ensure_ready()
    sess._goto_position(qt, secs=3.0)
trip = sess.run(duration=a.duration)
print("position hold restored" + (" (%s)" % trip if trip else ""))
if a.test_push > 0 and law.log:
    t, amp, q = (np.array([r[0] for r in law.log]),
                 np.array([r[1] for r in law.log]),
                 np.array([r[2] for r in law.log]))
    q0 = q[0]
    print("\nvirtual fingertip along J1's leverage direction:")
    for j in range(6):
        m = np.abs(np.degrees(q[:, j] - q0[j])) > 3.0
        print("  J%d: %s" % (j + 1,
              "follows at %.2f N" % amp[m.argmax()] if m.any() else
              "never moved (max %.1f N)" % amp[-1]))

