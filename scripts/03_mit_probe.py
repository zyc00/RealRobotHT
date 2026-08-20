"""MIT torque bring-up, ramped.

Run only from the folded park pose, where J2/J3 sit ~5 deg from their end stops,
so a complete loss of torque drops the arm 5 deg onto hard stops.

Ramps the gravity feedforward from 10% to 100% and records how the arm responds
at each level, which tells us both the sign convention and what t_ref actually
means in physical units.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm
from piper_ht.model import PiperModel
from piper_ht.calibration import TORQUE_CONST, BIAS, SCALE

ABORT_DEG = 9.0
KD = 1.0
KP = 0.0
DWELL = 1.5
GAINS = [0.10, 0.25, 0.50, 0.75, 1.00]

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
arm = PiperArm().connect(0.5)
mdl = PiperModel()

q0 = arm.q()
print("park pose (deg):", np.degrees(q0))
if np.degrees(q0)[1] > 20 or np.degrees(q0)[2] < -20:
    raise SystemExit("not in the folded park pose - run scripts/park.py first")
print("gravity model at this pose (N.m):", mdl.gravity_torque(q0))
print("abort if any joint moves more than %.0f deg\n" % ABORT_DEG)

log = []
aborted = False
try:
    for gain in GAINS:
        tau_ff = mdl.gravity_torque(arm.q()) * gain
        print("--- feedforward gain %.2f  t_ref = %s" % (gain, tau_ff.round(3)))
        t0 = time.time()
        while time.time() - t0 < DWELL:
            arm.mit_mode()
            q = arm.q()
            tau_ff = mdl.gravity_torque(q) * gain
            for i in range(6):
                arm.mit(i, q0[i], 0.0, KP, KD, tau_ff[i])
            d = np.degrees(np.abs(q - q0))
            log.append((gain, time.time() - t0, q.copy(), arm.current().copy(), arm.effort().copy()))
            if d.max() > ABORT_DEG:
                print("  !! ABORT: joint %d moved %.1f deg" % (d.argmax() + 1, d.max()))
                aborted = True
                break
            time.sleep(0.005)
        if aborted:
            break
        q = arm.q()
        cur = arm.current()
        print("    drift (deg): %s" % np.degrees(q - q0).round(2))
        print("    current (A): %s" % cur.round(3))
        print("    implied torque (N.m): %s" % (cur * TORQUE_CONST).round(3))
finally:
    # Always hand control back to the position loop, holding wherever we ended.
    q_end = arm.q()
    for _ in range(30):
        arm.move_j(q_end, speed_pct=10)
        time.sleep(0.01)
    print("\nreturned to position hold at", np.degrees(arm.q()).round(2))

np.savez("data/mit_probe.npz",
         gain=np.array([r[0] for r in log]),
         t=np.array([r[1] for r in log]),
         q=np.array([r[2] for r in log]),
         current=np.array([r[3] for r in log]),
         effort=np.array([r[4] for r in log]))
print("saved data/mit_probe.npz (%d samples)" % len(log))
