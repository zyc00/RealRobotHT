"""Gravity model fitted to the real arm, in SDK reported-effort units.

The URDF is right about the big links but wrong at the wrist, and the per-joint
current->torque scale is only known for J2/J3.  Fitting directly in reported
effort sidesteps both problems: gamma_j absorbs the unknown scale, and the fit
corrects the wrist geometry.  Everything downstream (deadbands, damping) is then
expressed in the same reported-effort units.

Measured 5-fold cross-validated RMS, reported-effort N.m (80-pose set with J1
and J6 exercised):

    J1 0.101   J2 0.256   J3 0.116   J4 0.473   J5 0.161   J6 0.056

Compare against the arm's own repeatability, measured by revisiting 10 poses
from unrelated directions:

    J1 0.005   J2 0.041   J3 0.022   J4 0.010   J5 0.018   J6 0.015

The hardware is 6-47x more repeatable than our model is accurate, so the
deadband is limited by MODEL error, not by friction or sensing.  J4 is the worst
offender and a generic trig basis fits it 2.5x better than the physical one,
which says the URDF wrist kinematics are wrong rather than the joint being
sticky.  More poses plus a richer model should close most of this gap.
"""

import os

import numpy as np

from .model import PiperModel

FIT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "grav_fit.npz")

# 3x the cross-validated residual: below this we cannot distinguish a real push
# from model error, so it sets the minimum usable deadband.
CV_RMS = np.array([0.101, 0.256, 0.116, 0.473, 0.161, 0.056])
MIN_DEADBAND = 3 * CV_RMS

# The worst gravity torque anywhere in the workspace is ~11.9 N.m (J2, fully
# extended), so a prediction beyond this is a broken model, not a real load.
MAX_PLAUSIBLE_EFFORT = 16.0


class GravityFit:
    def __init__(self, path=FIT_PATH, model=None):
        self.mdl = model or PiperModel()
        d = np.load(path, allow_pickle=True)
        self.links = [str(x) for x in d["links"]]
        self.scale = d["scale"]              # (6,)  URDF gravity scale
        self.dev = d["dev"]                  # (6,P) learned deviation
        self.const = d["const"]              # (6,)

    def effort(self, q, clamp=True):
        """Predicted holding effort (SDK units) with no external contact.

        Anchored to the scaled URDF model, so parameter directions the
        identification never excited degrade to the URDF rather than diverging.
        `clamp` is a last-resort guard: no joint on this arm can need more than
        ~16 N.m of gravity torque, so anything beyond that is a model failure
        and must never reach the controller.
        """
        Y = self.mdl.gravity_regressor(q, self.links)
        base = self.scale * self.mdl.gravity_torque(q)
        out = base + np.einsum("jp,jp->j", Y, self.dev) + self.const
        if clamp:
            out = np.clip(out, -MAX_PLAUSIBLE_EFFORT, MAX_PLAUSIBLE_EFFORT)
        return out

    def is_plausible(self, q):
        """False when the raw model output is physically impossible."""
        return bool(np.abs(self.effort(q, clamp=False)).max() <= MAX_PLAUSIBLE_EFFORT)
