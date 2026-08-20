"""Measured torque-sensing calibration for this specific arm.

Identified on 2026-08-19 from scripts/02_calibrate_torque.py (J2 sweep,
0-80 deg at J3=-90 deg, both directions, 18 static samples).  Re-measured after
an arm power cycle and reproduced to within 0.3% (J2 0.4157 -> 0.4144,
J3 0.3504 -> 0.3506), so these are properties of the hardware, not of a
particular session.

BIAS is only meaningful while the arm is energised; with motors disabled after a
clean restart every joint reads exactly 0.000 A.  Controllers should re-zero at
startup rather than trust these numbers (see piper_ht.admittance).

The SDK converts motor current to "effort" with a single fixed coefficient per
joint group (1.18125 N.m/A for J1-3, 0.95844 for J4-6, see
piper_msgs/msg_v2/feedback/arm_feedback_high_spd.py).  Regressing that reported
effort against the URDF gravity model gives a slope well below 1, i.e. the SDK
under-reports true joint torque.  The URDF model is the trustworthy side here:
its total mass (4.67 kg) matches the published arm mass, and a hand check of the
J2 holding torque at the folded pose agrees with it to ~10%.

    tau_true[j] = effort_reported[j] / SCALE[j]
                = current[A] * TORQUE_CONST[j]

Joints 1 and 6 have no gravity signature in any pose (their axes stay parallel
to gravity), so their scale cannot be identified this way; they are left at the
SDK default and only their bias is recorded.
"""

import numpy as np

# Slope of reported effort vs modelled gravity torque.
SCALE = np.array([1.0, 0.4144, 0.3506, 1.0, 1.0, 1.0])

# Fit quality, for reference: R^2 and rms residual in reported-effort N.m.
FIT_R2 = np.array([np.nan, 0.98727, 0.99832, np.nan, np.nan, np.nan])
FIT_RMS = np.array([np.nan, 0.1852, 0.0245, np.nan, np.nan, np.nan])

# Effective joint-side torque constants, N.m per amp of motor current.
SDK_COEFF = np.array([1.18125, 1.18125, 1.18125, 0.95844, 0.95844, 0.95844])
TORQUE_CONST = SDK_COEFF / SCALE      # -> [1.181, 2.842, 3.371, 0.958, 0.958, 0.958]

# Zero-load offset in reported effort (N.m), measured with the arm energised and
# holding poses where the modelled gravity torque is ~0.
BIAS = np.array([0.0565, -0.2081, 0.0426, 0.2500, 0.0330, -0.0099])

# Coulomb friction seen as the half-difference between up and down sweeps, in
# reported-effort N.m.  These came out very small, but the position controller
# was holding each pose for ~2 s, which can mask stiction, so treat them as a
# lower bound rather than a measurement.
FRICTION_LB = np.array([0.003, 0.014, 0.003, 0.005, 0.003, 0.009])

# J4 carries up to ~1 N.m of measured torque across poses where the URDF model
# predicts ~0. Unexplained; the wrist model is not trusted to better than ~1 N.m.
KNOWN_DISCREPANCY_J4 = 1.0


def true_torque(effort_reported):
    """Convert SDK-reported effort (N.m) to estimated true joint torque (N.m)."""
    return (np.asarray(effort_reported) - BIAS) / SCALE
