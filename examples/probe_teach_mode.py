"""What does the firmware's teach mode actually do?

Teach mode blocks commands but keeps streaming feedback, so we can watch the
drivers.  The motors are clearly not off - the arm does not collapse - so the
question is how much of the gravity load they carry, and where the rest goes.

Three numbers, in increasing order of usefulness:

  1. corr(measured effort, modelled gravity torque).  Near +1 means the drivers
     are pushing in proportion to gravity, i.e. there IS a compensator running.
  2. The OLS slope of measured on modelled, with an intercept.  The intercept
     absorbs the sensor bias, so unlike a ratio-of-means this is not corrupted
     by the pose mix you happened to sweep.  Slope 1.0 = full compensation.
  3. The effective end mass.  If the shortfall came from a global gain or a
     wrong installation_pos every joint would be short by the SAME factor.  It
     is not - the deficit grows toward the wrist - which is the signature of
     mass missing from the far end of the firmware's model.  So we fit the one
     free parameter that produces exactly that signature: a scale k on the
     gripper assembly (gripper_base + fingers, 0.50 kg in the URDF).

Only J2 and J3 have an identified torque scale (piper_ht.calibration.SCALE), so
the end-mass fit uses those two.  J4-J6 are reported for shape only.

Put the arm in TEACH MODE (press the button on the arm), then run this and move
it slowly through several poses - the wider the sweep, the tighter the fit.
"""
import sys
import time

import numpy as np
sys.path.insert(0, ".")
from piper_ht.calibration import SCALE
from piper_ht.model import PiperModel
from piperx_teleop import PiperArm

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
OUT = "data/teach_probe.npz"
GRIPPER_LINKS = ("gripper_base", "link7", "link8")


def model_without_gripper():
    """Same model with the gripper assembly massless, keeping the COM entries."""
    m = PiperModel()
    for name in GRIPPER_LINKS:
        _, com = m.links[name]
        m.links[name] = (0.0, com)
    return m


arm = PiperArm("can0").connect(require_control=False)
mdl = PiperModel()
bare = model_without_gripper()
gripper_mass = sum(mdl.links[n][0] for n in GRIPPER_LINKS)
np.set_printoptions(precision=2, suppress=True, floatmode="fixed")

mode = arm.control_mode()
print("control mode: %s" % mode)
if "TEACHING" not in mode:
    print("\n!! NOT in teach mode. Press the teach button on the arm, then re-run.")
    print("   (running anyway, so you can compare against normal position hold)\n")

print("move the arm slowly through several poses for %.0f s...\n" % SECONDS)
Q, E, C, T = [], [], [], []
t0 = time.time()
while time.time() - t0 < SECONDS:
    Q.append(arm.q())
    E.append(arm.effort())
    C.append(np.array([m.current for m in arm._motors()]) * 1e-3)
    T.append(time.time() - t0)
    time.sleep(0.02)

Q, E, C, T = np.array(Q), np.array(E), np.array(C), np.array(T)
# Model torques in SDK effort units, so they sit alongside the raw readings.
G = np.array([mdl.gravity_torque(q) for q in Q]) * SCALE
G0 = np.array([bare.gravity_torque(q) for q in Q]) * SCALE   # gripper removed

np.savez(OUT, q=Q, effort=E, current=C, t=T, model=G, model_bare=G0, mode=mode)

moved = np.degrees(Q.max(0) - Q.min(0))
print("joint range covered (deg):", moved.round(0))
if moved.max() < 5:
    print("  (barely moved - every fit below is weak; sweep it further next time)")

print()
print("  joint   measured effort      gravity model       slope    R^2    corr")
slope = np.full(6, np.nan)
for j in range(6):
    g, e = G[:, j], E[:, j]
    if g.std() > 1e-6:
        A = np.column_stack([g, np.ones_like(g)])
        (s, b), *_ = np.linalg.lstsq(A, e, rcond=None)
        resid = e - A @ np.array([s, b])
        r2 = 1.0 - resid.var() / e.var() if e.var() > 0 else np.nan
        c = np.corrcoef(e, g)[0, 1]
        slope[j] = s
    else:
        s = b = r2 = c = np.nan
    print("  J%d    %6.2f +/- %4.2f     %6.2f +/- %4.2f    %6.2f  %5.2f  %+.2f"
          % (j + 1, e.mean(), e.std(), g.mean(), g.std(), s, r2, c))
print("    slope 1.0 = the drivers carry the full modelled load; 0.0 = passive")

# --- how much end mass does the firmware think it has? ---------------------
# On the joints with an identified scale, ask what multiple of the gripper mass
# reconciles the reading with the model, assuming the rest of the arm is
# compensated in full:  E/SCALE - tau_bare = k * (tau_full - tau_bare)
FIT = [1, 2]                                   # J2, J3
x = (G[:, FIT] - G0[:, FIT]).ravel()           # gripper's own contribution
y = (E[:, FIT] - G0[:, FIT]).ravel()
k = float(x @ y / (x @ x)) if x @ x > 0 else np.nan
lever = np.abs(G[:, FIT] - G0[:, FIT]).mean()
print()
print("effective end mass (fit on J2,J3): k = %.2f  ->  %.2f kg of the %.2f kg gripper"
      % (k, k * gripper_mass, gripper_mass))
print("  the gripper moves %.2f N.m of J2/J3 torque on average in this sweep" % lever)
if lever < 0.05:
    print("  (too little gripper leverage in these poses - k is not meaningful)")
elif k < 0.35:
    print("  -> the firmware is compensating as if nothing were mounted at the wrist.")
    print("     Try declaring it:  ArmParamEnquiryAndConfig(end_load_param_setting_effective=0xAE,")
    print("                                                 set_end_load=0x01)   # half load")
elif k > 0.8:
    print("  -> the end mass IS in the firmware's model; the shortfall is elsewhere.")

print()
print("residual you must fight (SDK N.m): %s" % np.abs(E - G).mean(0).round(2))
print("current draw (A):                  %s" % np.abs(C).mean(0).round(2))
print("saved raw sweep -> %s" % OUT)
