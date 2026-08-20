"""Live camera preview for tuning camera pose.

Shows the full sensor view alongside the exact 256x256 image your policy will
see, so you frame against what is actually trained on rather than the raw view.

    python tools/camera_preview.py                    # 640x480 @ 20 fps, crop to 256
    python tools/camera_preview.py --res 1280 720
    python tools/camera_preview.py --train-res 224

  s  save both views to data/cam/     c  crop mode: centre / squash
  g  overlay: full / crosshair / none  f  flip      +/-  exposure
  q  quit
"""
import argparse
import os
import time

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--res", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
ap.add_argument("--fps", type=int, default=20,
                help="20 divides a 100 Hz control loop exactly (1 frame per 5 ticks)")
ap.add_argument("--train-res", type=int, default=256, help="policy input size")
ap.add_argument("--flip", action="store_true")
a = ap.parse_args()

cap = cv2.VideoCapture(a.device, cv2.CAP_V4L2)
if not cap.isOpened():
    raise SystemExit("cannot open /dev/video%d" % a.device)
# MJPG BEFORE the resolution: in YUYV the C920 drops to 10 fps at 720p and 5 at
# 1080p, and setting size first can leave the format pinned.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.res[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.res[1])
cap.set(cv2.CAP_PROP_FPS, a.fps)

T = a.train_res
print(__doc__)
os.makedirs("data/cam", exist_ok=True)
overlay, squash, flip, shots = 0, False, a.flip, 0
t_prev, fps = time.time(), 0.0


def to_train(img):
    """Exactly what the recorder should produce for the policy."""
    if squash:
        return cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0:y0 + s, x0:x0 + s], (T, T), interpolation=cv2.INTER_AREA)


while True:
    ok, frame = cap.read()
    if not ok:
        print("frame grab failed")
        break
    if flip:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now = time.time()
    fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
    t_prev = now

    view = frame.copy()
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    if not squash:
        # everything OUTSIDE this box is discarded before the policy sees it
        dark = view.copy()
        cv2.rectangle(dark, (0, 0), (w, h), (0, 0, 0), -1)
        view = cv2.addWeighted(view, 0.55, dark, 0.45, 0)
        view[y0:y0 + s, x0:x0 + s] = frame[y0:y0 + s, x0:x0 + s]
        cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), (0, 200, 255), 2)
    if overlay < 2:
        c = (0, 255, 0)
        cv2.line(view, (w // 2, y0), (w // 2, y0 + s), c, 1)
        cv2.line(view, (x0, h // 2), (x0 + s, h // 2), c, 1)
        cv2.circle(view, (w // 2, h // 2), 16, c, 1)
    if overlay == 0:
        for i in (1, 2):
            cv2.line(view, (x0 + s * i // 3, y0), (x0 + s * i // 3, y0 + s), (90, 90, 90), 1)
            cv2.line(view, (x0, y0 + s * i // 3), (x0 + s, y0 + s * i // 3), (90, 90, 90), 1)

    train = to_train(frame)
    big = cv2.resize(train, (s, s), interpolation=cv2.INTER_NEAREST)
    pane = np.zeros((max(h, s) + 34, w + s + 12, 3), np.uint8)
    pane[34:34 + h, 0:w] = view
    pane[34:34 + s, w + 12:w + 12 + s] = big
    cv2.putText(pane, "sensor %dx%d  %.1f fps  mode=%s" % (w, h, fps, "squash" if squash else "crop"),
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(pane, "policy sees %dx%d (shown enlarged)" % (T, T),
                (w + 18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    cv2.imshow("camera preview - s save, c crop mode, g overlay, q quit", pane)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        break
    elif k == ord("s"):
        cv2.imwrite("data/cam/full_%03d.png" % shots, frame)
        cv2.imwrite("data/cam/train_%03d.png" % shots, train)
        print("saved data/cam/full_%03d.png and train_%03d.png" % (shots, shots))
        shots += 1
    elif k == ord("c"):
        squash = not squash
    elif k == ord("g"):
        overlay = (overlay + 1) % 3
    elif k == ord("f"):
        flip = not flip
    elif k in (ord("+"), ord("=")):
        cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) + 10)
    elif k == ord("-"):
        cap.set(cv2.CAP_PROP_EXPOSURE, cap.get(cv2.CAP_PROP_EXPOSURE) - 10)

cap.release()
cv2.destroyAllWindows()
print("%d snapshot pair(s) in data/cam/" % shots)
