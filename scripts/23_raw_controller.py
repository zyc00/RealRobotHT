"""Raw Quest controller readout. No arm, no mapping, no frames.

Prints the right controller's position exactly as the headset reports it, and
how far it has travelled since you pressed ENTER.  If this does not track your
hand, nothing downstream can.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from robovr.quest3 import Quest3Server

srv = Quest3Server(port=7777, adb_reverse=True)
srv.start()
print("waiting for headset...")
st = srv.wait_for_state(timeout_s=10.0)
for _ in range(100):
    st = srv.latest()
    if st is not None and st.connected:
        break
    time.sleep(0.1)
if st is None or not st.connected:
    sys.exit("not connected")

print("connected.\n")
print("Hold the RIGHT controller. Press ENTER, then move it a KNOWN distance")
print("(e.g. 30 cm to your left along a table edge) and watch the numbers.")
input(">>> ENTER to start ")

p0 = None
maxd = 0.0
t0 = time.time()
last = 0.0
try:
    while time.time() - t0 < 40:
        st = srv.latest()
        if st is None:
            time.sleep(0.02); continue
        g = st.right_grip
        if g is None:
            if time.time() - last > 1.0:
                print("  right_grip is None - controller not reporting"); last = time.time()
            time.sleep(0.05); continue
        p = np.asarray(g.position, float)
        if p0 is None:
            p0 = p.copy()
        d = p - p0
        maxd = max(maxd, np.linalg.norm(d))
        if time.time() - last > 0.25:
            last = time.time()
            print("  pos %s m | moved %6.1f mm | max %6.1f mm | valid=%s flags=0x%X squeeze=%.2f"
                  % (p.round(3), np.linalg.norm(d) * 1000, maxd * 1000,
                     getattr(g, "valid", "?"), int(getattr(st, "right_grip_flags", 0)),
                     float(getattr(st, "right_squeeze", 0.0))))
except KeyboardInterrupt:
    pass
print("\nmax travel observed: %.0f mm" % (maxd * 1000))
print()
if maxd < 0.10:
    print("VERDICT: the headset is NOT tracking the controller.")
    print("  Quest 3 controllers have no tracking ring - they are tracked by the")
    print("  headset's cameras.  Out of view, the IMU dead-reckons a smooth, valid,")
    print("  full-rate pose that does not follow your hand.")
    print("  Fix: wear the headset, or set it FACING your working volume 0.5-1.5 m away.")
else:
    print("VERDICT: tracking looks healthy (%.0f mm of travel seen)." % (maxd * 1000))
print("If you moved 300 mm and this says ~80 mm, the headset is not tracking the")
print("controller properly - most likely it is out of the cameras' view.")
srv.close()
