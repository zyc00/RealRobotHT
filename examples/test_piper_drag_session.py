from types import SimpleNamespace
import unittest
from unittest.mock import patch

from piper_drag_session import PiperDragSession


class StartupTests(unittest.TestCase):
    def run_start(self, mode=6, freeze=False, status_frozen=False, delay=0):
        clock = [100.0]
        frames = []
        def stamp():
            return 99 if freeze or clock[0] < 100+delay else clock[0]-.001
        session = object.__new__(PiperDragSession)
        session.arm = SimpleNamespace(obs_time=stamp)
        session.piper = SimpleNamespace(
            MotionCtrl_2=lambda *args: frames.append(args),
            GetArmStatus=lambda: SimpleNamespace(time_stamp=99 if status_frozen else stamp(),
                                                 arm_status=SimpleNamespace(mode_feed=mode)))
        def sleep(dt):
            clock[0] += dt
        with patch('piper_drag_session.time.time', side_effect=lambda:clock[0]), \
             patch('piper_drag_session.time.monotonic', side_effect=lambda:clock[0]), \
             patch('piper_drag_session.time.sleep', side_effect=sleep):
            result = session._enter_mit(timeout=.5)
        self.assertEqual(frames, [(1, 6, 0, 0xAD, 0, 0)])
        return result, clock[0]-100

    def test_waits_for_post_switch_stable_feedback(self):
        result, elapsed = self.run_start(delay=.2)
        self.assertTrue(result)
        self.assertGreaterEqual(elapsed, .25)

    def test_old_joint_or_mode_feedback_cannot_start(self):
        self.assertFalse(self.run_start(freeze=True)[0])
        self.assertFalse(self.run_start(status_frozen=True)[0])

    def test_position_mode_cannot_start_torque_loop(self):
        self.assertFalse(self.run_start(mode=1)[0])
