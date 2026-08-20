"""Log external-torque estimates while a human pushes the arm.

The arm stays in position hold throughout, so this is safe; we are only
watching how hard the position loop has to fight back.

    tau_ext = (reported_effort - bias)/scale  -  gravity_model  -  startup_zero

The startup zero is captured from the first second, during which nobody should
be touching the arm.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.model import PiperModel
from piper_ht.calibration import SCALE, BIAS

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")

arm = PiperArm().connect(0.5)
mdl = PiperModel()
q_hold = arm.q()

def tau_true():
    return (arm.effort() - BIAS) / SCALE

print("holding at", np.degrees(q_hold).round(2))
print("capturing zero (do not touch the arm)...")
t0 = time.time(); Z = []
while time.time() - t0 < 1.5:
    arm.move_j(q_hold, speed_pct=10)
    Z.append(tau_true() - mdl.gravity_torque(arm.q()))
    time.sleep(0.01)
zero = np.array(Z).mean(0)
noise = np.array(Z).std(0)
print("zero  :", zero.round(3))
print("noise :", noise.round(4), " <- detection floor\n")
print("PUSH THE ARM NOW - %.0f s" % DUR)

log = []
t0 = time.time()
while time.time() - t0 < DUR:
    arm.move_j(q_hold, speed_pct=10)
    q = arm.q()
    ext = tau_true() - mdl.gravity_torque(q) - zero
    log.append((time.time() - t0, q.copy(), ext.copy(), arm.dq().copy()))
    time.sleep(0.005)

T = np.array([r[0] for r in log]); Q = np.array([r[1] for r in log])
E = np.array([r[2] for r in log]); D = np.array([r[3] for r in log])
np.savez("data/sense.npz", t=T, q=Q, ext=E, dq=D, zero=zero, noise=noise)

print("\ncaptured %d samples over %.1f s" % (len(T), T[-1]))
print("\nexternal torque estimate, per joint:")
print("  joint     min      max   max|.|   noise   SNR")
for j in range(6):
    snr = np.abs(E[:, j]).max() / max(noise[j], 1e-6)
    print("  j%d   %7.3f %7.3f %8.3f %7.4f %6.1f" % (j+1, E[:,j].min(), E[:,j].max(), np.abs(E[:,j]).max(), noise[j], snr))
print("\nposition deviation while pushed (deg): ", np.degrees(np.abs(Q - q_hold).max(0)).round(3))
print("peak joint velocity seen (rad/s):      ", np.abs(D).max(0).round(3))
