"""Identify the gravity model from the real arm.

Visits a spread of poses, holds each one still, and records the torque the
position loop needs to hold it.  Then fits, per joint, a model linear in the
links' barycentric parameters:

    reported_effort_j(q) ~= Y_j(q) @ gamma_j + c_j

gamma_j absorbs the unknown per-joint current->torque scale, so this is a
predictive model in reported-effort units - exactly what the admittance loop
needs to subtract.  It replaces the URDF's wrist parameters, which are wrong by
~2.4 N.m on J4.

NOBODY MAY TOUCH THE ARM while this runs, and the workspace must be clear.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.arm import PiperArm, JOINT_LIMITS
from piper_ht.model import PiperModel

N_POSES = int(sys.argv[1]) if len(sys.argv) > 1 else 70
N_REPEAT = 10          # poses revisited later, to measure repeatability directly
SPEED, SETTLE, SAMPLE = 18, 1.2, 0.5
MIN_Z, MAX_R, MARGIN_DEG = 0.25, 0.55, 8.0

np.set_printoptions(precision=3, suppress=True, floatmode="fixed")
mdl = PiperModel()
rng = np.random.default_rng(7)

def ok(q):
    m = np.degrees(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q))
    if m.min() < MARGIN_DEG:
        return False
    T = mdl.fk(q)
    p = T[:3, 3]
    return p[2] >= MIN_Z and np.hypot(p[0], p[1]) <= MAX_R

poses = []
while len(poses) < N_POSES:
    # J1 and J6 must move: holding them at 0 last time left the model's
    # dependence on them completely unconstrained, and it extrapolated to
    # thousands of N.m once the arm was dragged off J6=0.
    q = np.radians([rng.uniform(-60, 60),
                    rng.uniform(10, 90),
                    rng.uniform(-140, -30),
                    rng.uniform(-80, 80),
                    rng.uniform(-50, 50),
                    rng.uniform(-90, 90)])
    if ok(q):
        poses.append(q)

# Greedy nearest-neighbour ordering, to cut travel time.
order, remaining = [0], list(range(1, len(poses)))
while remaining:
    last = poses[order[-1]]
    nxt = min(remaining, key=lambda i: np.abs(poses[i] - last).sum())
    order.append(nxt); remaining.remove(nxt)
poses = [poses[i] for i in order]

# Revisit a handful of poses at the end, reached from unrelated directions, so
# the fit residual can be compared against genuine repeatability.
repeat_idx = rng.choice(len(poses), size=min(N_REPEAT, len(poses)), replace=False)
repeats = [poses[i].copy() for i in repeat_idx]
rng.shuffle(repeats)
poses = poses + repeats
is_repeat = np.array([False] * (len(poses) - len(repeats)) + [True] * len(repeats))

zs = [mdl.fk(q)[2, 3] for q in poses]
print("%d poses (%d of them repeat visits) | tool height %.2f-%.2f m | est. runtime ~%.0f s"
      % (len(poses), int(is_repeat.sum()), min(zs), max(zs), len(poses) * 4.0))
print("!! J1 now sweeps +/-60 deg, so the arm swings SIDE TO SIDE around its base.")
print("!! Clear the whole area around the arm, not just in front of it.")
input("\n>>> Workspace clear and HANDS OFF? ENTER to start ")

arm = PiperArm().connect(0.5)
if not all(arm.is_enabled()):
    q = arm.q(); arm.piper.EnableArm(7)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.move_j(q, speed_pct=10); time.sleep(0.01)

Q, E = [], []
t_start = time.time()
for n, q in enumerate(poses):
    t0 = time.time()
    while time.time() - t0 < 10:
        arm.move_j(q, speed_pct=SPEED)
        if np.abs(arm.q() - q).max() < np.radians(0.7):
            break
        time.sleep(0.01)
    time.sleep(SETTLE)
    qs, es = [], []
    t0 = time.time()
    while time.time() - t0 < SAMPLE:
        arm.move_j(q, speed_pct=SPEED)
        qs.append(arm.q()); es.append(arm.effort())
        time.sleep(0.01)
    Q.append(np.mean(qs, 0)); E.append(np.mean(es, 0))
    if n % 5 == 0 or n == len(poses) - 1:
        print("  %2d/%d  q=%s  eff=%s" % (n + 1, len(poses),
              np.degrees(Q[-1]).round(0), E[-1].round(2)))

Q, E = np.array(Q), np.array(E)
np.savez("data/ident.npz", q=Q, effort=E, is_repeat=is_repeat)
print("\ncollected %d poses in %.0f s -> data/ident.npz" % (len(Q), time.time() - t_start))
