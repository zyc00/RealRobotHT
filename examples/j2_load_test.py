"""Does J2's stiction depend on LOAD (arm configuration), not just J2 angle?

J2 carries everything outboard. Extending/folding J3 changes the torque J2's
transmission must hold at a FIXED J2 angle - i.e. the load its bearings/gears
carry - without moving J2 itself. If friction is load-dependent, J2's breakaway
torque rises with that load; if it is intrinsic to J2's angle, it stays flat.

  fix J2 (default 45 deg), vary J3 over a set spanning J2 load 0.05..2.4 N.m.
  At each config, measure J2 breakaway BOTH directions -> F_s = (u+ + u-)/2
  (cancels the gravity torque, leaves friction). Plot F_s vs the model's
  |gravity torque about J2| (the load). Positive slope = load-dependent.

  Control: --j1-control also varies J1, which does NOT change J2 load - F_s
  should stay flat there. A flat J1 control + rising J3 curve is the clean proof.

GENTLE: detects breakaway at 0.5 deg and immediately catches J2 with a firm
damped re-pin, so the arm barely twitches (no 11 deg slip, no gravity swing).

    python examples/j2_load_test.py                      # J2=45, J3 load sweep
    python examples/j2_load_test.py --j2 40 --reps 3     # average 3 per config
    python examples/j2_load_test.py --j1-control
"""
import argparse
import os
import sys
import time

_PIPERCTL = os.path.expanduser("~/miniforge3/envs/piperctl/bin/python")
try:
    from piperx_teleop import MitCommand, PiperModel, TorqueSession, require_patched_sdk
    require_patched_sdk()
except (RuntimeError, ImportError):
    if os.path.exists(_PIPERCTL) and os.path.realpath(sys.executable) != os.path.realpath(_PIPERCTL):
        os.execv(_PIPERCTL, [_PIPERCTL] + sys.argv)
    raise

import numpy as np

sys.setswitchinterval(0.0005)
RAD = np.pi / 180.0
JT = 1                       # test joint index (J2)
PIN_KP, PIN_KD = 4.0, 1.0
CATCH_KD = 3.0              # firm damping to arrest the joint the instant it moves
RATE = 0.10                 # N.m/s ramp - gentle
PRELOAD = 0.4               # N.m (below J2 friction ~1.2)
BREAK_DEG = 0.5             # detect early -> minimal motion
JUMP_S = 0.12               # brief - just enough to confirm, then catch
MIN_Z, MAX_X = 0.08, 0.45

ap = argparse.ArgumentParser()
ap.add_argument("--j2", type=float, default=45.0, help="fixed J2 angle, deg")
ap.add_argument("--j3-list", type=float, nargs="+", default=[-70, -60, -50, -90, -100],
                help="J3 angles (span J2 load); measured in this order")
ap.add_argument("--j1-control", action="store_true", help="also vary J1 (load-invariant control)")
ap.add_argument("--reps", type=int, default=2, help="breakaway reps per direction per config")
ap.add_argument("--cap", type=float, default=2.6)
ap.add_argument("--can", default="can0")
ap.add_argument("--out", default="data/j2_load_test.npz")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()
mdl = PiperModel()
LINKS = [k for k in mdl._walk(np.zeros(6))[0].keys() if k not in ("world", "base_link")]


def safe(q):
    T, _ = mdl._walk(q)
    return all(T[k][2, 3] >= MIN_Z and T[k][0, 3] <= MAX_X for k in LINKS)


class LoadLaw:
    def __init__(self):
        self.mode = "pin"
        self.q_ref = None
        self.u = 0.0
        self.direction = 1.0
        self.catch = False

    def __call__(self, s):
        if self.q_ref is None:
            self.q_ref = s.q.copy()
        tau = mdl.gravity_torque(s.q)
        kp = np.full(6, PIN_KP)
        kd = np.full(6, PIN_KD)
        if self.catch:
            kd[JT] = CATCH_KD                      # arrest the test joint
        if self.mode == "ramp":
            kp[JT] = kd[JT] = 0.0
            tau[JT] += self.u * self.direction
        return MitCommand(t_ff=tau, p_des=self.q_ref, kp=kp, kd=kd)


law = LoadLaw()
sess = TorqueSession(law, can=a.can, hz=200.0)


def goto(q_target, secs=2.0):
    law.mode = "pin"; law.catch = False
    q0 = law.q_ref.copy() if law.q_ref is not None else sess.q()
    t0 = time.time()
    while time.time() - t0 < secs + 0.3:
        f = min((time.time() - t0) / secs, 1.0)
        law.q_ref = q0 + f * (q_target - q0)
        time.sleep(0.01)


def gentle_breakaway(direction):
    """Ramp J2 from preload; catch at BREAK_DEG so the arm barely moves."""
    q_pin = sess.q()[JT]
    law.direction = direction
    law.u = PRELOAD
    law.catch = False
    law.mode = "ramp"
    t0 = time.time()
    ub = np.nan
    while True:
        time.sleep(0.004)
        law.u = min(PRELOAD + RATE * (time.time() - t0), a.cap)
        if (sess.q()[JT] - q_pin) * direction > BREAK_DEG * RAD:
            ub = law.u
            break
        if law.u >= a.cap or not sess.running or not safe(sess.q()):
            break
    # immediate firm catch, then settle back to the station
    law.mode = "pin"; law.catch = True; law.q_ref = sess.q()
    time.sleep(JUMP_S)
    law.catch = False
    return ub


def measure_config(label, q_config, load):
    goto(q_config, secs=2.5)
    time.sleep(0.5)
    ups, ums = [], []
    for _ in range(a.reps):
        goto(q_config, secs=0.8); time.sleep(0.3)
        ups.append(gentle_breakaway(+1.0))
        goto(q_config, secs=0.8); time.sleep(0.3)
        ums.append(gentle_breakaway(-1.0))
    up, um = np.nanmean(ups), np.nanmean(ums)
    fs = np.nanmean([up, um])
    print("  %-14s load %.2f N.m ->  F_s %.2f  (+ %.2f / - %.2f)  N.m" % (label, load, fs, up, um))
    return fs, up, um


# build config list: (label, q, load)
configs = []
for j3 in a.j3_list:
    q = np.radians([0.0, a.j2, j3, 0.0, 10.0, 0.0])
    if not safe(q):
        print("skip J3=%.0f (clearance)" % j3); continue
    configs.append(("J3=%+.0f" % j3, q, abs(mdl.gravity_torque(q)[JT])))
ctrl = []
if a.j1_control:
    for j1 in [-40.0, 0.0, 40.0]:
        q = np.radians([j1, a.j2, -70.0, 0.0, 10.0, 0.0])
        if safe(q):
            ctrl.append(("J1=%+.0f" % j1, q, abs(mdl.gravity_torque(q)[JT])))

print("J2 load-dependence: J2 fixed at %.0f deg, %d J3 configs, %d reps/dir" % (
    a.j2, len(configs), a.reps))
print("gentle: catch at %.1f deg, ramp %.2f N.m/s" % (BREAK_DEG, RATE))
if not a.yes:
    input(">>> CLEAR WORKSPACE, hand on E-STOP. ENTER to start ")

rows = []
with sess:
    goto(configs[0][1], secs=3.0)
    print("\n-- J3 load sweep (J2 fixed) --")
    for label, q, load in configs:
        fs, up, um = measure_config(label, q, load)
        rows.append((label, load, fs, up, um))
    ctrl_rows = []
    if ctrl:
        print("\n-- J1 control (load-invariant) --")
        for label, q, load in ctrl:
            fs, up, um = measure_config(label, q, load)
            ctrl_rows.append((label, load, fs, up, um))
    goto(np.radians([0.0, a.j2, -70.0, 0.0, 10.0, 0.0]), secs=3.0)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
loads = np.array([r[1] for r in rows]); fss = np.array([r[2] for r in rows])
np.savez(a.out, j2=a.j2, labels=[r[0] for r in rows], load=loads, F_s=fss,
         Fs_plus=[r[3] for r in rows], Fs_minus=[r[4] for r in rows],
         ctrl_labels=[r[0] for r in ctrl_rows], ctrl_load=[r[1] for r in ctrl_rows],
         ctrl_Fs=[r[2] for r in ctrl_rows])

print("\n" + "=" * 56)
ok = np.isfinite(fss)
if ok.sum() >= 3:
    slope, icpt = np.polyfit(loads[ok], fss[ok], 1)
    r = np.corrcoef(loads[ok], fss[ok])[0, 1]
    print("J2 F_s vs load:  slope %.3f N.m per N.m load   r %.2f" % (slope, r))
    print("  intrinsic (zero-load) F_s %.2f N.m,  spans %.2f..%.2f over load range" % (
        icpt, fss[ok].min(), fss[ok].max()))
    if slope > 0.05 and r > 0.6:
        print("  -> LOAD-DEPENDENT: J2 stiction rises with configuration load (q matters).")
    elif abs(slope) < 0.05 or r < 0.4:
        print("  -> LOAD-INDEPENDENT: J2 stiction ~ intrinsic to J2 angle (q barely matters).")
    else:
        print("  -> weak/mixed load dependence.")
if ctrl_rows:
    cf = np.array([r[2] for r in ctrl_rows])
    print("J1 control (load fixed): F_s %.2f ± %.2f N.m  (flat = confirms it's LOAD, not pose)" % (
        np.nanmean(cf), np.nanstd(cf)))
print("saved %s" % a.out)
print("position hold restored" + (" (%s)" % sess.trip if sess.trip else ""))
