"""Align the teleop frame to YOUR body, by measuring one deliberate motion.

Everything else checks out - robot axes verified physically, mapping verified
against the log, firmware convention confirmed.  What is left is the yaw between
the frame we project onto and the direction you actually face.  Rather than
infer it from the headset (which points wherever it is set down, and which
follows your gaze rather than your body), measure it: push your hand straight
away from yourself and we compute the offset that makes that be "forward".
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from piper_ht.quest_teleop import QuestCartesianSource, QuestTeleopConfig

cfg = QuestTeleopConfig()
cfg.frame_mode = "latched"
src = QuestCartesianSource(cfg).start()
print("waiting for the headset...")
for _ in range(200):
    if src.poll().connected:
        break
    time.sleep(0.1)
if not src.poll().connected:
    sys.exit("Quest not connected")

print("\nStand/sit exactly as you will when teleoperating.")
input(">>> ENTER, then SQUEEZE the grip and push your hand STRAIGHT AWAY from your\n"
      "    chest about 30 cm, and release the grip while it is still out there. ")

world = np.zeros(3)
t0 = time.time()
seen = False
while time.time() - t0 < 15:
    s = src.poll()
    if s.clutch:
        seen = True
        world += s.dpos_world
    elif seen and np.linalg.norm(world) > 0.05:
        break
    time.sleep(0.005)

n = np.linalg.norm(world)
print("\nraw controller displacement (Quest world): %s mm, |%.0f| mm"
      % ((world * 1000).round(0), n * 1000))
if n < 0.05:
    sys.exit("too little motion - re-run and push further, releasing at full extension")

horiz = np.array([world[0], 0.0, world[2]])
if np.linalg.norm(horiz) < 0.03:
    sys.exit("that motion was almost entirely vertical - push horizontally, away from your chest")
horiz /= np.linalg.norm(horiz)

# Frame forward must equal this direction.  heading = atan2(-fx, -fz).
heading = np.degrees(np.arctan2(-horiz[0], -horiz[2]))
print("your 'forward' is at heading %.1f deg in the Quest world frame" % heading)
np.savez("data/yaw_calib.npz", heading_deg=heading, forward=horiz)
print("saved data/yaw_calib.npz")
print("\nteleop will now use a FIXED frame aimed along your body, not the headset.")
src.stop()
