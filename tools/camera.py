"""Threaded camera capture for teleop recording.

Two measured problems make a naive `cap.read()` unusable in a control loop:

  * read() BLOCKS until the next frame - 33.5 ms on 50/50 ticks at 640x480/30.
    Called inline, a 100 Hz control loop collapses to camera rate.
  * The V4L2 driver queues frames (BUFFERSIZE=4, ~3 in practice). Whenever the
    consumer pauses, the next read() returns a frame up to 100 ms stale, and
    nothing in the API tells you it is old.

So a background thread grabs continuously and keeps only the newest frame with
the timestamp it arrived. The control loop takes that latest frame and gets its
age, never blocking and never silently receiving stale data.
"""

import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, device=0, res=(640, 480), fps=20, train_res=256, squash=False,
                 fourcc="MJPG"):
        # 20 fps divides the 100 Hz control loop exactly (one new frame every 5
        # ticks), which keeps frame-to-tick alignment regular. 30 fps gives 3.33
        # ticks per frame, so frames land irregularly against ticks.
        self.device, self.res, self.fps = device, res, fps
        self.train_res, self.squash = train_res, squash
        self.fourcc = fourcc
        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._seq = 0
        self._dropped = 0

    def start(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("cannot open /dev/video%d" % self.device)
        # FOURCC before size: in YUYV the C920 drops to 10 fps at 720p, and
        # setting the size first can leave the format pinned.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # advisory; we drain anyway
        for _ in range(10):
            cap.read()                            # let auto-exposure settle
        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < 5.0:
            time.sleep(0.01)
        if self._frame is None:
            raise RuntimeError("no frame within 5 s")
        return self

    def _run(self):
        while not self._stop.is_set():
            ok = self._cap.grab()
            t = time.time()
            if not ok:
                time.sleep(0.005)
                continue
            ok, f = self._cap.retrieve()
            if not ok:
                continue
            with self._lock:
                if self._frame is not None:
                    self._dropped += 1        # the consumer never took the last one
                self._frame, self._stamp, self._seq = f, t, self._seq + 1

    def latest(self):
        """(frame, capture_time, age_seconds, seq). Never blocks."""
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0.0, 0
            return self._frame, self._stamp, time.time() - self._stamp, self._seq

    def train_image(self, frame=None):
        """The exact array the policy sees: square crop (or squash) at train_res."""
        if frame is None:
            frame, _, _, _ = self.latest()
        if frame is None:
            return None
        T = self.train_res
        if self.squash:
            return cv2.resize(frame, (T, T), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return cv2.resize(frame[y0:y0 + s, x0:x0 + s], (T, T), interpolation=cv2.INTER_AREA)

    def stats(self):
        with self._lock:
            return dict(seq=self._seq, dropped=self._dropped)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
