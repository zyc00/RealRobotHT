"""Balanced drag: gravity comp + per-joint friction fill so all six joints
feel like the same virtual joint.

People suffer from friction IMBALANCE (heavy geared shoulder vs light wrist),
not friction itself. Identified on this arm (examples/friction_id.py):

    kinetic friction f0:  J1 0.32  J2 0.60  J3 0.29 | J4 0.09  J5 0.07  (N.m)
    apparent inertia  M:  J1 0.15  J2 1.1   J3 0.32 | J4 0.03  J5 0.006 (kg m^2)

The law adds, per joint, a velocity-gated fill toward a common target level:

    tau = G(q) + alpha * max(f0_j - F_TARGET, 0) * tanh(qdot_j / V_EPS)

Safety by construction: the fill only acts while the joint MOVES, is bounded
by alpha * (f0 - target) < f0, and therefore can never exceed the friction it
opposes - no self-driving, no chatter (tanh, not sign). Breakaway (static
friction) is deliberately untouched: it cannot be cancelled by velocity
feedback, only masked by dither, which buzzes.

    python examples/balanced_drag.py                # alpha 0.8
    python examples/balanced_drag.py --alpha 0.5    # milder
    python examples/balanced_drag.py --alpha 0      # = plain drag mode (A/B!)
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

# Kinetic friction per joint, N.m in command units. Per-direction minima from
# data/friction_fit.npz, manually sanity-screened: J3's falling-direction and
# all of J6's fits were degenerate (too-short traces) and are NOT used - J3
# takes its clean rising-direction value, J6 gets no fill (it is light anyway).
F0 = np.array([0.31, 0.60, 0.29, 0.07, 0.06, 0.00])
F_TARGET = 0.08          # the wrist's level: what every joint should feel like
V_EPS = 0.08             # rad/s; fill ramps in smoothly over this speed

ap = argparse.ArgumentParser(description="drag mode with balanced friction feel")
ap.add_argument("--alpha", type=float, default=0.8,
                help="fraction of the excess friction to fill (0 = plain drag)")
ap.add_argument("--duration", type=float, default=0.0)
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

mdl = PiperModel()
FILL = a.alpha * np.maximum(F0 - F_TARGET, 0.0)


def law(s):
    return mdl.gravity_torque(s.q) + FILL * np.tanh(s.qdot / V_EPS)


sess = TorqueSession(law, can=a.can)
print("pose (deg):", np.degrees(sess.q()).round(1))
print("friction fill (N.m):", FILL.round(2), " (alpha %.2f, target %.2f)" % (a.alpha, F_TARGET))
if not a.yes:
    input(">>> ENTER to go compliant - joints should feel UNIFORM while moving ")
trip = sess.run(duration=a.duration)
print("position hold restored" + (" (%s)" % trip if trip else ""))
