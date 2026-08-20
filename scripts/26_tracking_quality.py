"""Score Quest 3 controller tracking for a given headset placement.

Quest 3 controllers have no tracking ring - they are tracked optically by the
headset's cameras, with the IMU only covering brief gaps.  Out of view, tracking
fails SILENTLY: the pose stays valid and updates at full rate while dead
reckoning, so the only way to know is to measure.

Two tests, neither needing external equipment:

  1. STILL   - hold the controller still.  Good tracking is quiet; dead
               reckoning drifts.
  2. OUT-AND-BACK - move out ~30 cm and return to the SAME physical spot.
               Good tracking returns to near zero; dead reckoning does not.

Run once per candidate placement and compare the scores.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from robovr.quest3 import Quest3Server

label = " ".join(sys.argv[1:]) or "unnamed placement"
srv = Quest3Server(port=7777, adb_reverse=True)
srv.start()
print("waiting for headset...")
for _ in range(150):
    st = srv.latest()
    if st is not None and st.connected and st.right_grip is not None:
        break
    time.sleep(0.1)
st = srv.latest()
if st is None or not st.connected or st.right_grip is None:
    sys.exit("not connected / no controller pose - is the app running and the controller on?")


def grab(seconds):
    out, t0 = [], time.time()
    while time.time() - t0 < seconds:
        s = srv.latest()
        if s is not None and s.right_grip is not None:
            out.append(np.asarray(s.right_grip.position, float))
        time.sleep(0.008)
    return np.array(out)


print("\n=== placement: %s ===" % label)
input(">>> Hold the controller STILL where you normally work, then press ENTER ")
still = grab(5.0)
drift = np.linalg.norm(still[-1] - still[0]) * 1000
jitter = np.linalg.norm(still - still.mean(0), axis=1).std() * 1000
rate = len(still) / 5.0
print("  drift over 5 s : %6.1f mm      (good < 5)" % drift)
print("  jitter         : %6.2f mm      (good < 2)" % jitter)
print("  samples/s      : %6.0f" % rate)

input("\n>>> Now move it OUT about 30 cm and back to the SAME spot, then ENTER ")
print("  (recording for 8 s - do the out-and-back now)")
trip = grab(8.0)
travel = np.linalg.norm(trip - trip[0], axis=1).max() * 1000
ret = np.linalg.norm(trip[-1] - trip[0]) * 1000
print("  peak travel    : %6.1f mm      (should be ~300)" % travel)
print("  return error   : %6.1f mm      (good < 20)" % ret)

score = []
score.append(("still drift", drift < 5))
score.append(("jitter", jitter < 2))
score.append(("travel registered", travel > 200))
score.append(("returns to start", ret < 20))
print("\n  %-20s %s" % ("check", "result"))
for n, okv in score:
    print("  %-20s %s" % (n, "PASS" if okv else "FAIL"))
good = sum(1 for _, o in score if o)
print("\nVERDICT for '%s': %d/4" % (label, good))
if good == 4:
    print("  Tracking is healthy here - usable placement.")
elif travel < 200:
    print("  The headset is NOT seeing the controller (only %.0f mm of %.0f mm registered)."
          % (travel, 300))
    print("  Move the headset so its CAMERAS FACE your hands.")
else:
    print("  Marginal - tracking registers motion but is drifting or noisy.")
    print("  Try closer (0.6-1.2 m), or angled more directly at your working volume.")
srv.close()
