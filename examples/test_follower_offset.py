import unittest
import numpy as np

from joint_safety import JointGuard, common_limits
from position_follower import PositionFollower
from mit_follower import MitFollower
from startup_position import entry_target
from test_position_follower import FakeArm


class OffsetTests(unittest.TestCase):
    def setUp(self):
        self.offset = np.radians([0,0,0,0,0,45])
        self.fg = JointGuard(common_limits())
        self.lg = JointGuard(common_limits(common_limits()-self.offset[:,None]))
        self.leader = FakeArm('can0', [0,1,-1,0,0,0])
        self.arm = FakeArm('can1', [0,1,-1,0,0,0])

    def make(self, cls):
        return cls(self.leader,self.arm,offset=self.offset,guard=self.lg,follower_guard=self.fg)

    def test_mapping_and_leader_upper_limit(self):
        for cls in (PositionFollower, MitFollower):
            f = self.make(cls)
            np.testing.assert_allclose(f._target(self.leader.pose),self.leader.pose+self.offset)
            q=self.leader.pose.copy();q[5]=np.radians(75)
            with self.assertRaises(RuntimeError): f._target(q)
        np.testing.assert_allclose(np.degrees(self.lg.hard[5]),[-119.00028,73.99972],atol=.001)

    def test_actual_follower_pose_is_not_offset_again(self):
        f=self.make(MitFollower)
        self.arm.pose[5]=np.radians(100)
        f.command=self.arm.pose.copy()
        f._check_actual(self.arm.pose)
        np.testing.assert_array_equal(entry_target(self.arm.pose,self.fg),self.arm.pose)

    def test_negative_offset_mapping_and_limits(self):
        self.offset = -self.offset
        self.lg = JointGuard(common_limits(common_limits()-self.offset[:,None]))
        for cls in (PositionFollower, MitFollower):
            f = self.make(cls)
            self.assertAlmostEqual(f._target(self.leader.pose)[5], -np.pi/4)
            q = self.leader.pose.copy(); q[5] = np.radians(-75)
            with self.assertRaises(RuntimeError):
                f._target(q)
        np.testing.assert_allclose(np.degrees(self.lg.hard[5]), [-73.99972,119.00028], atol=.001)

    def test_offset_is_slewed_not_jumped_and_log_keeps_raw_leader(self):
        f=self.make(PositionFollower);f.log_enabled=True
        f.start()
        f.tick(monotonic=f.previous+.02)
        self.assertAlmostEqual(f.command[5],.01)
        self.assertAlmostEqual(f.telem['mapped_target'][5],np.pi/4)
        self.assertEqual(f.rows[-1][6],0.)


if __name__ == '__main__':
    unittest.main()
