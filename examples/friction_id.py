"""Identify each joint's kinetic friction curve: terminal-velocity sweeps.

Physics: in torque mode with gravity fed forward, add a known extra torque u
on ONE joint and record the whole trajectory. The joint obeys

    M * dv/dt = u - f0 - b*v        (apparent inertia, Coulomb + viscous friction)

so regressing acceleration on [u, 1, v] identifies M, f0 and b per joint and
direction - no torque sensor involved, the label is the command. (A pure
terminal-velocity protocol fails on the wrist: kinetic friction there is so
small the joint never plateaus inside its runway - measured, J5 crossed 60 deg
still accelerating. The trajectory fit uses exactly that data instead.)
The smallest u that produces motion brackets the breakaway (static friction).

Everything runs in ONE TorqueSession (no mode churn): a stateful law pins all
joints with a firmware PD while repositioning, then frees the test joint with
gravity + u while the others stay pinned. The runtime's watchdogs stay active
throughout.

Output data/friction_id.npz feeds examples/balanced_drag.py, which fills in
alpha * f(qdot) per joint so all six feel like the same virtual joint.

The arm moves BY ITSELF across wide joint ranges: CLEAR THE WORKSPACE.

    python examples/friction_id.py                # all joints, ~8-10 min
    python examples/friction_id.py --joints 5     # just J5 (smoke test)
"""
import argparse
import os
import sys
import time

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

RAD = np.pi / 180.0
PIN_KP, PIN_KD = 3.0, 0.8
V_CAP = np.array([1.2, 1.0, 1.2, 2.5, 3.0, 2.0])     # stop escalating past this, rad/s
TRIAL_S = 3.5
MARGIN = np.radians(20)
LIMITS = np.array([[-2.618, 2.618], [0.0, 3.14], [-2.967, 0.0],
                   [-1.553, 1.553], [-1.553, 1.553], [-2.0944, 2.0944]])

# per joint: base pose (deg), runway (start_for_+u, start_for_-u, deg on that joint),
# torque levels to escalate through (N.m, command units)
PLAN = {
    0: (dict(), (-55.0, 55.0), [0.15, 0.25, 0.4, 0.6, 0.9, 1.3]),
    1: (dict(), (18.0, 95.0),  [0.8, 1.4, 2.0, 2.6, 3.2]),   # J2 static ~3 N.m
    2: (dict(), (-100.0, -22.0), [0.2, 0.35, 0.55, 0.8, 1.1, 1.5]),
    3: (dict(), (-58.0, 58.0), [0.15, 0.25, 0.4, 0.6, 0.9]),
    4: (dict(), (-58.0, 58.0), [0.1, 0.18, 0.3, 0.45, 0.65]),
    5: (dict(), (-85.0, 85.0), [0.06, 0.10, 0.14, 0.18, 0.24]),  # tiny inertia
}
BASE = np.radians([0.0, 45.0, -70.0, 0.0, 10.0, 0.0])

ap = argparse.ArgumentParser()
ap.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                help="1-indexed joints to identify")
ap.add_argument("--can", default="can0")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

mdl = PiperModel()


class IDLaw:
    """Pin everything; optionally free one joint with gravity + extra torque."""

    def __init__(self):
        self.q_ref = None          # pin target (updated by the main thread)
        self.test_j = None         # None = all pinned
        self.u = 0.0

    def __call__(self, s):
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        tau = mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        if self.test_j is not None:
            j = self.test_j
            kp[j] = 0.0
            kd[j] = 0.0
            tau[j] += self.u
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


law = IDLaw()
sess = TorqueSession(law, can=a.can)


def goto(q_target, secs=2.0):
    """Slide the pin target; the in-session PD tracks it."""
    law.test_j = None
    q0 = law.q_ref if law.q_ref is not None else sess.q()
    t0 = time.time()
    while time.time() - t0 < secs + 0.4:
        f = min((time.time() - t0) / secs, 1.0)
        law.q_ref = q0 + f * (q_target - q0)
        time.sleep(0.01)
    time.sleep(0.3)


def trial(j, u):
    """Free joint j with extra torque u; return (v_terminal, moved, clipped)."""
    q_ref = sess.q().copy()
    law.q_ref = q_ref
    law.u = u
    law.test_j = j                 # atomically frees the joint
    ts, qs = [], []
    t0 = time.time()
    clipped = False
    while (dt := time.time() - t0) < TRIAL_S:
        q = sess.q()
        ts.append(dt)
        qs.append(q[j])
        # runway check in the DIRECTION OF TRAVEL only (a start pose may
        # legitimately sit near the limit behind it)
        ahead = (LIMITS[j, 1] - q[j]) if u > 0 else (q[j] - LIMITS[j, 0])
        if ahead < MARGIN:
            clipped = True
            break
        if len(qs) > 6 and abs(qs[-1] - qs[-6]) / max(ts[-1] - ts[-6], 1e-3) > V_CAP[j]:
            clipped = True          # fast enough - stop gently, data is plenty
            break
        time.sleep(0.008)
    law.test_j = None              # re-pin at the CURRENT pose, not the start
    law.q_ref = sess.q().copy()
    ts, qs = np.array(ts), np.array(qs)
    moved = len(ts) > 8 and abs(qs[-1] - qs[0]) > np.radians(2.0)
    v_peak = float(np.abs(np.gradient(qs, ts)).max()) if moved else 0.0
    return v_peak, moved, clipped, ts, qs


if not a.yes:
    input(">>> Friction ID for joints %s: the arm moves BY ITSELF across wide "
          "ranges. Workspace clear? ENTER " % a.joints)

rows = []      # (joint, u_signed, v_term, moved, clipped)
sess.start()
try:
    for jn in a.joints:
        j = jn - 1
        _, (start_pos, start_neg), levels = PLAN[j]
        for direction, start_deg in ((+1, start_pos), (-1, start_neg)):
            hits = 0
            for u in levels:
                pose = BASE.copy()
                pose[j] = np.radians(start_deg)
                goto(pose)
                if not sess.running:
                    raise RuntimeError("session tripped: %s" % sess.trip)
                v_peak, moved, clipped, ts, qs = trial(j, direction * u)
                rows.append(dict(j=j, u=direction * u, moved=moved,
                                 clipped=clipped, t=ts, q=qs))
                print("  J%d  u=%+5.2f  ->  peak v %+6.3f rad/s%s"
                      % (jn, direction * u, v_peak * np.sign(direction),
                         "" if moved else "  (no motion)"))
                if moved:
                    hits += 1
                if hits >= 3:
                    break
finally:
    sess.stop()

try:                                    # keep other joints' earlier trials
    _old = np.load("data/friction_id.npz")
    _n = int(_old["meta"][0])
    redone = {jn - 1 for jn in a.joints}
    for i in range(_n):
        if int(_old["trial_%d_j" % i]) not in redone:
            rows.append(dict(j=int(_old["trial_%d_j" % i]),
                             u=float(_old["trial_%d_u" % i]),
                             moved=bool(_old["trial_%d_moved" % i]),
                             clipped=False,
                             t=_old["trial_%d_t" % i], q=_old["trial_%d_q" % i]))
except FileNotFoundError:
    pass
np.savez("data/friction_id.npz",
         meta=np.array([len(rows)]),
         **{"trial_%d_%s" % (i, k): np.asarray(v) for i, r in enumerate(rows)
            for k, v in (("j", r["j"]), ("u", r["u"]), ("moved", r["moved"]),
                         ("t", r["t"]), ("q", r["q"]))})
print("\nsaved %d trials -> data/friction_id.npz" % len(rows))

# ---- fit  M*dvdt = u - f0 - b*v  per joint & direction ----------------------
print("\n  joint dir   M_app (kg m^2)   f0 (N.m)   b (N.m s)   breakaway <=")
fits = {}
for j in range(6):
    for direction in (+1, -1):
        A_rows, y_rows = [], []
        breakaway = None
        for r in rows:
            if r["j"] != j or np.sign(r["u"]) != direction:
                continue
            if not r["moved"]:
                breakaway = max(breakaway or 0.0, abs(r["u"]))
                continue
            t, q = r["t"], r["q"]
            if len(t) < 25:
                continue
            v = np.gradient(q, t)
            # light smoothing before the second derivative
            k = np.ones(7) / 7.0
            vs = np.convolve(v, k, mode="same")
            acc = np.gradient(vs, t)
            m = np.abs(vs) > 0.06          # moving samples only
            m[:5] = m[-5:] = False
            A_rows.append(np.column_stack([np.full(m.sum(), r["u"]),
                                           np.ones(m.sum()), vs[m]]))
            y_rows.append(acc[m])
        if not A_rows:
            continue
        A = np.vstack(A_rows); y = np.concatenate(y_rows)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)   # [1/M, -f0/M, -b/M]
        if abs(c[0]) < 1e-6:
            continue
        M = 1.0 / c[0]
        f0, b = -c[1] * M, -c[2] * M
        fits[(j, direction)] = (M, f0, b)
        print("   J%d   %s   %10.4f   %8.3f   %8.3f      %s"
              % (j + 1, "+" if direction > 0 else "-", M, f0, b,
                 "%.2f" % breakaway if breakaway else "< smallest level"))
np.savez("data/friction_fit.npz",
         keys=np.array([[j, d] for (j, d) in fits]),
         vals=np.array([fits[k] for k in fits]))
print("fit saved -> data/friction_fit.npz")
