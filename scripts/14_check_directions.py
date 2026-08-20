"""Verify the controller -> robot axis mapping. Never commands the arm.

Guides you through three motions and reports which robot axis each one drove,
so the signs are measured rather than assumed.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.quest_teleop import QuestCartesianSource, QuestTeleopConfig

AXES = ["x (forward)", "y (left)", "z (up)"]
TESTS = [
    ("push the controller AWAY from you", 0, +1),
    ("lift the controller UP", 2, +1),
    ("move the controller to YOUR LEFT", 1, +1),
]

cfg = QuestTeleopConfig(yaw_offset_deg=float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
src = QuestCartesianSource(cfg).start()
print("waiting for the headset...")
for _ in range(200):
    if src.poll().connected:
        break
    time.sleep(0.1)
s = src.poll()
if not s.connected:
    sys.exit("Quest not connected")
print("connected. frame heading %.0f deg\n" % src.frame.heading_deg())

results = []
for label, axis, want in TESTS:
    input(">>> ENTER, then SQUEEZE the grip and %s (~20 cm), then release. " % label.upper())
    total = np.zeros(3)
    world = np.zeros(3)
    path = 0.0
    t0 = time.time()
    seen = False
    clutch_samples = 0
    drops = 0
    was = False
    while time.time() - t0 < 12:
        s = src.poll()
        if s.clutch:
            seen = True
            clutch_samples += 1
            total += s.dpos
            world += s.dpos_world
            path += float(np.linalg.norm(s.dpos_world))
        else:
            if was and seen:
                drops += 1
            if seen and np.linalg.norm(total) > 0.02:
                break
        was = s.clutch
        time.sleep(0.005)
    r, u, f = src.frame.axes()
    print("   raw controller motion (Quest world) = %s mm, path length %.0f mm"
          % ((world * 1000).round(1), path * 1000))
    print("   frame axes: right %s  up %s  fwd %s" % (r.round(2), u.round(2), f.round(2)))
    print("   clutched samples %d, clutch drops %d%s"
          % (clutch_samples, drops, "   <-- clutch flickered, motion lost" if drops > 1 else ""))
    if not seen or np.linalg.norm(total) < 0.025:
        print("   NET motion only %.0f mm over a %.0f mm path - too little to judge."
              % (np.linalg.norm(total) * 1000, path * 1000))
        if path > 2.5 * max(np.linalg.norm(total), 1e-6):
            print("   The path is much longer than the net, so you came back before releasing.")
        print("   Redo this one: move ONE way and release the grip while still out there.\n")
        results.append((label, None, None, total))
        continue
    dom = int(np.argmax(np.abs(total)))
    ok = (dom == axis) and (np.sign(total[dom]) == want)
    print("   robot delta = %s mm -> dominant %s, sign %+d   %s\n"
          % ((total * 1000).round(1), AXES[dom], int(np.sign(total[dom])),
             "OK" if ok else "*** MISMATCH ***"))
    results.append((label, dom, int(np.sign(total[dom])), total))

print("\n================ summary ================")
allok = True
inconclusive = 0
for (label, axis, want), (lb, dom, sg, tot) in zip(TESTS, results):
    if dom is None:
        print("  %-38s inconclusive (too little motion)" % label)
        inconclusive += 1
        continue
    ok = (dom == axis) and (sg == want)
    allok &= ok
    print("  %-38s -> %-12s %+d  %s" % (label, AXES[dom], sg, "OK" if ok else "MISMATCH"))
if allok and inconclusive == 0:
    print("\nMapping verified on all three axes - safe to run live.")
elif allok:
    print("\n%d axis/axes inconclusive (too little motion), but every axis that DID"
          " register was correct." % inconclusive)
    print("Re-run just to confirm, moving further and releasing at full extension.")
else:
    print("\nMapping is wrong. Fixes:")
    print("  - all three rotated among each other -> wrong --yaw-offset")
    print("  - one axis inverted -> flip that sign in QuestTeleopConfig")
    print("    (forward_sign / right_sign / up_sign)")
src.stop()
