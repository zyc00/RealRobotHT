import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from piperx_teleop.arm import PiperArm
from position_follower import PositionFollower, fresh_q, prepare_feedback


class FakeArm:
    move_j = PiperArm.move_j
    clamp = PiperArm.clamp
    _install_pos = PiperArm._install_pos

    def __init__(self, name, pose):
        self.can_name = name
        self.pose = np.array(pose, float)
        self.stamp = time.time()
        self.mode = 1
        self.enabled = True
        self.frames = []
        # Deliberately no torque/MIT/gripper API: tests fail if any is used.
        self.piper = SimpleNamespace(MotionCtrl_2=lambda *args:self.frames.append(('mode', args)),
                                     JointCtrl=lambda *args:self.frames.append(('joint', args)),
                                     EnableArm=lambda *args:self.frames.append(('enable', args)))

    def obs_time(self):
        return self.stamp

    def q(self):
        return self.pose.copy()

    def is_enabled(self):
        return [self.enabled]*6

    def in_teach_mode(self):
        return False

    def status(self):
        return SimpleNamespace(mode_feed=self.mode)


class FollowerTests(unittest.TestCase):
    def test_live_speed_settings_apply_in_control_loop(self):
        leader = FakeArm('can0', [0, 1, -1, 0, 0, 0])
        arm = FakeArm('can1', [0, 1, -1, 0, 0, 0])
        follower = PositionFollower(leader, arm, log=True)
        follower.start()
        count = len(arm.frames)
        follower.set_speed('follower_speed', .2)
        follower.set_speed('follower_speed_pct', 30)
        self.assertEqual(len(arm.frames), count)  # HTTP setter sends no frames
        leader.pose[0] = .5
        follower.tick(monotonic=follower.previous+.02)
        self.assertAlmostEqual(follower.command[0], .004)
        self.assertEqual(arm.frames[-2], ('mode', (1, 1, 30, 0, 0, 0)))
        self.assertEqual(follower.speed_rows[-1], [.2, 30])
        modes = len([f for f in arm.frames if f[0]=='mode'])
        follower.tick(monotonic=follower.previous+.02)
        self.assertEqual(len([f for f in arm.frames if f[0]=='mode']), modes)

    def test_bad_live_speed_settings_rejected(self):
        follower = PositionFollower(FakeArm('can0', [0,1,-1,0,0,0]), FakeArm('can1', [0,1,-1,0,0,0]))
        for name, value in [('follower_speed', 0), ('follower_speed', 2.1),
                            ('follower_speed', float('nan')), ('follower_speed_pct', 0),
                            ('follower_speed_pct', 101), ('follower_speed_pct', 20.5)]:
            with self.assertRaises(ValueError):
                follower.set_speed(name, value)
        self.assertEqual(follower.speed_settings(), dict(follower_speed=.5, follower_speed_pct=20))

    def test_silent_follower_restores_feedback_without_motor_commands(self):
        arm = FakeArm('can1', [0, 1, -1, 0, 0, 0])
        arm.stamp = 0
        calls = []
        def restore(*args):
            calls.append(args)
            arm.stamp = time.time()
        arm.piper.MasterSlaveConfig = restore
        np.testing.assert_allclose(prepare_feedback(arm), arm.pose)
        self.assertEqual(calls, [(0xFC, 0, 0, 0)])
        self.assertEqual(arm.frames, [])
        prepare_feedback(arm)
        self.assertEqual(len(calls), 1)  # healthy stream is left alone

    def test_unrecoverable_feedback_fails_closed(self):
        arm = FakeArm('can1', [0, 1, -1, 0, 0, 0])
        arm.stamp = 0
        arm.piper.MasterSlaveConfig = lambda *args: None
        with self.assertRaises(RuntimeError):
            prepare_feedback(arm, timeout=0)
        self.assertEqual(arm.frames, [])

    def setup_pair(self):
        leader = FakeArm('can0', [.5, 1.2, -1.2, .2, -.2, .1])
        follower = FakeArm('can1', [0, 1, -1, 0, 0, 0])
        mirror = PositionFollower(leader, follower)
        mirror.start()
        return leader, follower, mirror

    def test_only_normal_position_frames_and_bounded_startup(self):
        leader, follower, mirror = self.setup_pair()
        start = mirror.command.copy()
        now = time.time()
        mirror.tick(now, mirror.previous+.02)
        self.assertLessEqual(np.max(abs(mirror.command-start)), .0100001)
        for name, args in follower.frames:
            if name == 'mode':
                self.assertEqual(args[:2], (1, 1))
                self.assertEqual(args[3], 0)  # normal position control, not MIT
                self.assertEqual(args[5], 0)  # do not reassert mounting configuration
        self.assertEqual(sum(kind == 'mode' for kind, _ in follower.frames), 1)
        joint = [args for kind, args in follower.frames if kind == 'joint'][-1]
        np.testing.assert_allclose(np.array(joint)*np.pi/180000, mirror.command, atol=1e-5)

    def test_reaches_absolute_leader_angles_without_offset(self):
        leader, follower, mirror = self.setup_pair()
        for _ in range(100):
            now = time.time()
            leader.stamp = follower.stamp = now
            mirror.tick(now, mirror.previous+.02)
            follower.pose = mirror.command.copy()
        np.testing.assert_allclose(mirror.command, leader.pose, atol=1e-8)

    def test_start_and_stop_preserve_parked_pose_without_limit_jump(self):
        leader = FakeArm('can0', [0, 1, -1, 0, 0, 0])
        follower = FakeArm('can1', [0, -.04, .04, 0, 0, 0])
        mirror = PositionFollower(leader, follower)
        mirror.start()
        first = [args for kind, args in follower.frames if kind == 'joint'][0]
        np.testing.assert_allclose(np.array(first)*np.pi/180000, follower.pose, atol=1e-5)
        with patch('position_follower.time.sleep'):
            mirror.stop()
        last = [args for kind, args in follower.frames if kind == 'joint'][-1]
        self.assertEqual(first, last)

    def test_stale_feedback_and_disabled_motor_stop_commands(self):
        for which in ('leader', 'follower', 'disabled'):
            leader, follower, mirror = self.setup_pair()
            if which == 'leader':
                leader.stamp -= 1
            elif which == 'follower':
                follower.stamp -= 1
            else:
                follower.enabled = False
            before = len(follower.frames)
            with self.assertRaises(RuntimeError):
                mirror.tick(time.time(), mirror.previous+.02)
            self.assertEqual(len(follower.frames), before)

    def test_mode_fault_and_stop_hold(self):
        leader, follower, mirror = self.setup_pair()
        follower.mode = 4
        mirror.tick(time.time(), mirror.previous+.02)
        with self.assertRaises(RuntimeError):
            mirror.tick(time.time(), mirror.previous+.6)
        with patch('position_follower.time.sleep'):
            mirror.stop()
        final = [args for kind, args in follower.frames if kind == 'joint'][-1]
        np.testing.assert_allclose(np.array(final)*np.pi/180000, follower.pose, atol=1e-5)
        self.assertFalse(mirror.active)

    def test_same_bus_and_limits_rejected(self):
        arm = FakeArm('can0', [0, 1, -1, 0, 0, 0])
        with self.assertRaises(ValueError):
            PositionFollower(arm, arm)
        leader, follower, mirror = self.setup_pair()
        leader.pose[0] = 10
        with self.assertRaises(RuntimeError):
            mirror.tick(time.time(), mirror.previous+.02)


if __name__ == '__main__':
    unittest.main()
