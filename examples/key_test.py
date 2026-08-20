"""Keyboard-only test. Touches no arm, no CAN, no config.

Prints every key the source receives and the displacement it accumulates.
If numbers move here, the keyboard is fine and the problem is downstream.
Press q to quit.
"""
import sys
import time

import numpy as np
from piperx_teleop import Config
from piperx_teleop.sources import KeyboardSource

src = KeyboardSource(Config()).start()
print("press keys: arrows, w s, u p i k j l, g, [ ] - =   (q quits)\r")
print("nothing prints until a key is received\r\n")
n = 0
try:
    prev_disp = src.disp.copy()
    prev_rot = src.R.as_rotvec().copy()
    prev_grip = src.gripper_closed
    while True:
        sample = src.poll()                 # reads one key and applies it
        changed = (not np.allclose(prev_disp, src.disp)
                   or not np.allclose(prev_rot, src.R.as_rotvec())
                   or prev_grip != src.gripper_closed
                   or sample.quit)
        if changed:
            n += 1
            sys.stdout.write("  #%-3d disp %s mm | rot %s deg | grip %s\r\n"
                             % (n, (src.disp * 1000).round(1),
                                np.degrees(src.R.as_rotvec()).round(1),
                                src.gripper_closed))
            sys.stdout.flush()
            prev_disp = src.disp.copy()
            prev_rot = src.R.as_rotvec().copy()
            prev_grip = src.gripper_closed
        if sample.quit:
            break
        time.sleep(0.005)
finally:
    src.stop()
    print("\r\nreceived %d keys" % n)
