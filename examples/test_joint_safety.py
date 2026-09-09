import time
import unittest
from types import SimpleNamespace

import numpy as np

from joint_safety import JointGuard, common_limits, read_limits, format_limits
from position_follower import PositionFollower, checked_target
from test_position_follower import FakeArm


class JointSafetyTests(unittest.TestCase):
    def setUp(self):
        self.guard = JointGuard(common_limits())
        self.q = self.guard.hard.mean(axis=1)

    def test_interior_unchanged(self):
        extra = np.arange(6)*0.1
        out, near = self.guard.apply(self.q, np.ones(6), extra)
        np.testing.assert_array_equal(out, extra)
        np.testing.assert_array_equal(near, np.zeros(6))

    def test_range_report_preserves_distinct_firmware_limits(self):
        leader = common_limits()
        follower = leader.copy()
        follower[5] = np.radians([-170, 170])
        report = format_limits(leader, follower, self.guard.hard, 'can0', 'can1')
        self.assertIn('Leader firmware (can0)', report)
        self.assertIn('Follower firmware (can1)', report)
        row = next(line for line in report.splitlines() if line.startswith('J6 '))
        self.assertIn('170.00', row)
        self.assertIn('120.00', row)
        self.assertIn('119.00', row)

    def test_each_boundary_resists_outward_motion(self):
        for j in range(6):
            for side, sign in ((0, -1), (1, 1)):
                q = self.q.copy()
                q[j] = self.guard.hard[j, side]-sign*self.guard.zone/2
                v, extra = np.zeros(6), np.zeros(6)
                v[j], extra[j] = sign*0.3, sign*0.1
                out, near = self.guard.apply(q, v, extra)
                self.assertLess(out[j]*sign, 0)
                self.assertAlmostEqual(near[j], 0.5)
                self.assertLessEqual(abs(out[j]-extra[j]*0.5), self.guard.cap[j]+1e-10)
                q[j] = self.guard.hard[j, side]
                with self.assertRaisesRegex(RuntimeError, 'J%d' % (j+1)):
                    self.guard.check(q)

    def test_invalid_inputs_and_empty_intersection(self):
        for q in ([0]*5, [np.nan]*6, [np.inf]*6):
            with self.assertRaises(RuntimeError):
                self.guard.check(q)
            with self.assertRaises(RuntimeError):
                checked_target(q)
        with self.assertRaises(ValueError):
            common_limits(np.tile([10., 11.], (6, 1)))

    def test_intersection_never_widens_model(self):
        model = common_limits()
        broad = np.tile([-4., 4.], (6, 1))
        np.testing.assert_array_equal(common_limits(broad), model)
        broad[3] = [-1., 1.]
        np.testing.assert_array_equal(common_limits(broad)[3], [-1., 1.])

    def test_query_requires_matching_fresh_replies(self):
        reply = SimpleNamespace(time_stamp=0, current_motor_angle_limit_max_vel=SimpleNamespace(
            motor_num=0, min_angle_limit=-900, max_angle_limit=900))
        def query(j, kind):
            self.assertEqual(kind, 1)
            reply.time_stamp = time.time()
            reply.current_motor_angle_limit_max_vel.motor_num = j
        arm = SimpleNamespace(can_name='fake', piper=SimpleNamespace(
            SearchMotorMaxAngleSpdAccLimit=query, GetCurrentMotorAngleLimitMaxVel=lambda: reply))
        np.testing.assert_allclose(read_limits(arm), np.tile([-np.pi/2, np.pi/2], (6, 1)))
        arm.piper.SearchMotorMaxAngleSpdAccLimit = lambda *args: None
        reply.time_stamp = 0
        with self.assertRaises(RuntimeError):
            read_limits(arm, timeout=0.01)

    def test_follower_rejects_guard_crossing_before_command(self):
        leader, arm = FakeArm('can0', self.q), FakeArm('can1', self.q)
        follower = PositionFollower(leader, arm, guard=self.guard)
        follower.start()
        count = len(arm.frames)
        leader.pose[3] = self.guard.hard[3, 1]
        with self.assertRaisesRegex(RuntimeError, 'J4'):
            follower.tick()
        self.assertEqual(len(arm.frames), count)

    def test_leader_start_outside_guard_does_not_enable_or_reposition(self):
        from piper_drag_session import PiperDragSession
        arm = FakeArm('can0', self.q)
        arm.enabled = False
        arm.pose[3] = self.guard.hard[3, 1]
        session = object.__new__(PiperDragSession)
        session.arm, session.piper = arm, arm.piper
        session.law = SimpleNamespace(guard=self.guard)
        with self.assertRaisesRegex(RuntimeError, 'J4'):
            session._ensure_ready()
        self.assertEqual(arm.frames, [])


if __name__ == '__main__':
    unittest.main()
