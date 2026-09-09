import unittest
from unittest.mock import patch

import numpy as np

from joint_safety import JointGuard, common_limits
from startup_position import entry_target, position_leader, position_arms
from test_position_follower import FakeArm
from piperx_teleop.arm import RAD2CMD


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.guard = JointGuard(common_limits())
        self.q = np.radians([-8, -2.7, 2.2, 36, 3, 26])

    def test_target_preserves_other_joints(self):
        np.testing.assert_allclose(np.degrees(entry_target(self.q, self.guard)), [-8, 3, -3, 36, 3, 26])
        q = self.q.copy()
        q[1] = np.radians(-6)
        with self.assertRaises(RuntimeError):
            entry_target(q, self.guard)

    def test_cancel_sends_nothing(self):
        arm = FakeArm('can0', self.q)
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            position_leader(arm, self.guard, confirm=lambda _: 'no')
        self.assertEqual(arm.frames, [])

    def test_soft_zone_pose_is_not_pushed_to_ten_degrees(self):
        q = np.radians([-8, 6, -6, 36, 3, 26])
        np.testing.assert_array_equal(entry_target(q, self.guard), q)

    def test_entry_has_clearance_from_all_hard_boundaries(self):
        for side in (0, 1):
            target = entry_target(self.guard.limits[:, side], self.guard)
            self.guard.check(target)
            np.testing.assert_allclose(abs(target-self.guard.hard[:, side]), np.radians(2))

    def test_already_interior_sends_nothing(self):
        arm = FakeArm('can0', entry_target(self.q, self.guard))
        position_leader(arm, self.guard, confirm=lambda _: self.fail('unnecessary prompt'))
        self.assertEqual(arm.frames, [])

    def run_fake(self, moving):
        arm = FakeArm('can0', self.q)
        clock = [1000.]
        arm.obs_time = lambda: clock[0]
        def send(*raw):
            arm.frames.append(('joint', raw))
            if moving:
                arm.pose = np.asarray(raw)/RAD2CMD
        arm.piper.JointCtrl = send
        def sleep(dt):
            clock[0] += dt
        with patch('startup_position.time.time', side_effect=lambda: clock[0]), \
             patch('startup_position.time.monotonic', side_effect=lambda: clock[0]), \
             patch('startup_position.time.sleep', side_effect=sleep):
            if moving:
                position_leader(arm, self.guard, confirm=lambda _: 'MOVE')
                self.guard.check(arm.pose)
                np.testing.assert_allclose(arm.pose, entry_target(self.q, self.guard), atol=np.radians(.5))
                commands = np.array([x[1] for x in arm.frames if x[0]=='joint'])/RAD2CMD
                self.assertLessEqual(np.max(abs(np.diff(commands, axis=0))), np.radians(.061))
            else:
                with self.assertRaisesRegex(RuntimeError, 'timed out'):
                    position_leader(arm, self.guard, confirm=lambda _: 'MOVE')
        self.assertEqual([x for x in arm.frames if x[0]=='mode'], [('mode', (1,1,10,0,0,0))])

    def test_success(self):
        self.run_fake(True)

    def test_stall_times_out(self):
        self.run_fake(False)

    def test_pair_moves_independently_and_holds_both_on_stall(self):
        for stalled in (False, True):
            arms = [FakeArm('can0', self.q), FakeArm('can1', self.q + np.radians([1, 0, 0, 0, 0, -20]))]
            targets = [entry_target(a.pose, self.guard) for a in arms]
            clock = [1000.]
            events = []
            for i, arm in enumerate(arms):
                arm.obs_time = lambda: clock[0]
                def send(*raw, i=i, arm=arm):
                    events.append((i, clock[0]))
                    if not (stalled and i == 1):
                        arm.pose = np.asarray(raw)/RAD2CMD
                arm.piper.JointCtrl = send
            def sleep(dt):
                clock[0] += dt
            with patch('startup_position.time.time', side_effect=lambda: clock[0]), \
                 patch('startup_position.time.monotonic', side_effect=lambda: clock[0]), \
                 patch('startup_position.time.sleep', side_effect=sleep):
                if stalled:
                    with self.assertRaisesRegex(RuntimeError, 'timed out'):
                        position_arms(arms, self.guard, confirm=lambda _: 'MOVE')
                else:
                    position_arms(arms, self.guard, confirm=lambda _: 'MOVE')
                    for arm, target in zip(arms, targets):
                        np.testing.assert_allclose(arm.pose, target, atol=np.radians(.5))
            self.assertEqual(events[0][0], 0)
            self.assertEqual(events[1][0], 1)
            self.assertEqual(events[0][1], events[1][1])
            self.assertEqual([e[0] for e in events[-2:]], [0, 1])

    def test_pair_cancel_and_invalid_second_arm_send_nothing(self):
        arms = [FakeArm('can0', self.q), FakeArm('can1', self.q)]
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            position_arms(arms, self.guard, confirm=lambda _: 'no')
        arms[1].pose[1] = np.radians(-20)
        with self.assertRaisesRegex(RuntimeError, 'manual recovery'):
            position_arms(arms, self.guard, confirm=lambda _: self.fail('invalid preview'))
        for arm in arms:
            self.assertEqual(arm.frames, [])


if __name__ == '__main__':
    unittest.main()
