"""Preview the camera through the SAME path the recorder uses.

tools/camera_preview.py does blocking cap.read() and is for framing the shot.
This example goes through tools.camera.Camera - the threaded, latest-frame
capture a control loop uses - so what you see is what a 100 Hz recording tick
would actually get: the newest frame, how stale it is, and how many frames the
grabber dropped because the consumer didn't keep up.

Pipeline: capture 640x480 @ 30 -> centre-crop 480x480 -> resize 256x256.
(The sensor has no native 480x480 mode; asking V4L2 for it returns 640x480,
so the square stage is the centre crop.)

    python examples/camera_preview.py                       # /dev/video0, 30 fps
    python examples/camera_preview.py --device 1
    python examples/camera_preview.py --hz 100              # poll at control rate

  left pane: 480x480 centre crop   right pane: train_image() (what the policy sees)
  s  save both views to data/cam/  c  toggle centre-crop / squash   q  quit
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from tools.camera import Camera

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--res", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--train-res", type=int, default=256)
ap.add_argument("--squash", action="store_true", help="resize instead of centre-crop")
ap.add_argument("--hz", type=float, default=30.0,
                help="poll rate; set 100 to mimic the control loop")
a = ap.parse_args()

print(__doc__)
os.makedirs("data/cam", exist_ok=True)
shots = 0
age_max = 0.0

with Camera(device=a.device, res=tuple(a.res), fps=a.fps,
            train_res=a.train_res, squash=a.squash) as cam:
    while True:
        tick = time.time()
        frame, stamp, age, seq = cam.latest()
        age_max = max(age_max, age)
        train = cam.train_image(frame)

        h, w = frame.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        square = frame[y0:y0 + s, x0:x0 + s]      # the 480x480 stage
        big = cv2.resize(train, (s, s), interpolation=cv2.INTER_NEAREST)
        pane = np.zeros((s + 34, 2 * s + 12, 3), np.uint8)
        pane[34:34 + s, 0:s] = square
        pane[34:34 + s, s + 12:s + 12 + s] = big
        st = cam.stats()
        cv2.putText(pane,
                    "crop %dx%d  seq %d  age %4.0f ms (max %4.0f)  dropped %d" %
                    (s, s, seq, age * 1e3, age_max * 1e3, st["dropped"]),
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(pane, "policy sees %dx%d" % (a.train_res, a.train_res),
                    (s + 18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        cv2.imshow("camera (recorder path) - s save, c crop mode, q quit", pane)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("s"):
            cv2.imwrite("data/cam/full_%03d.png" % shots, frame)
            cv2.imwrite("data/cam/train_%03d.png" % shots, train)
            print("saved data/cam/full_%03d.png and train_%03d.png" % (shots, shots))
            shots += 1
        elif k == ord("c"):
            cam.squash = not cam.squash

        # latest() never blocks, so pace the loop ourselves like a control tick
        time.sleep(max(0.0, 1.0 / a.hz - (time.time() - tick)))

cv2.destroyAllWindows()
print("%d snapshot pair(s) in data/cam/" % shots)
