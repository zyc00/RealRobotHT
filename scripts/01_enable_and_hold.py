"""First powered step: enable the arm and immediately pin it where it stands.

Logs joint motion throughout so any jump on enable is measured rather than
guessed at. Aborts to standby if the arm moves more than ABORT_DEG.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.model import PiperModel

ABORT_DEG = 8.0
HOLD_S = 3.0
SPEED = 10

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
mdl = PiperModel()

q0 = arm.q()
print("start pose (deg):", np.degrees(q0))
print("enable status   :", arm.is_enabled())
print("\nenabling and holding...")

arm.piper.EnableArm(7)
t0 = time.time()
peak = np.zeros(6)
traj = []
aborted = False
while time.time() - t0 < HOLD_S:
    arm.move_j(q0, speed_pct=SPEED)
    q = arm.q()
    d = np.degrees(np.abs(q - q0))
    peak = np.maximum(peak, d)
    traj.append((time.time() - t0, q.copy()))
    if d.max() > ABORT_DEG:
        print("!! ABORT: joint moved %.1f deg" % d.max())
        arm.standby()
        aborted = True
        break
    time.sleep(0.01)

print("enable status   :", arm.is_enabled())
print("peak excursion per joint (deg):", peak)
print("max excursion   : %.3f deg" % peak.max())

if not aborted:
    print("\nsettling, then sampling torque for 2 s at the held pose...")
    time.sleep(1.0)
    E, Q = [], []
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q0, speed_pct=SPEED)
        E.append(arm.effort()); Q.append(arm.q())
        time.sleep(0.01)
    E = np.array(E); Q = np.array(Q)
    q_mean = Q.mean(0)
    tau_meas = E.mean(0)
    tau_mdl = mdl.gravity_torque(q_mean)
    print("\npose (deg)        :", np.degrees(q_mean))
    print("measured tau (N.m):", tau_meas, " +/- ", E.std(0))
    print("model tau    (N.m):", tau_mdl)
    print("ratio meas/model  :", np.where(np.abs(tau_mdl) > 0.15, tau_meas / np.where(np.abs(tau_mdl) > 1e-9, tau_mdl, 1), np.nan))
    np.savez("data/hold_sample.npz", q=Q, effort=E)
    print("\nsaved data/hold_sample.npz")
print("\narm left ENABLED and holding position.")
