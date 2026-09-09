import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mit_follower import MitFollower
from test_position_follower import FakeArm
from joint_safety import JointGuard, common_limits
from b601_drag import ReferenceLaw, BalancedDrag, DynamicsAdapter
from test_b601_reference import FakeDynamics


class MitFollowerTests(unittest.TestCase):
    def setUp(self):
        self.clock = [1000.]
        self.leader = FakeArm('can0', [0, 1, -1, 0, 0, 0])
        self.arm = FakeArm('can1', [0, 1, -1, 0, 0, 0])
        for a in (self.arm, self.leader):
            a.obs_time = lambda: self.clock[0]
        self.arm.piper.JointMitCtrl = lambda *args: self.arm.frames.append(('mit', args))
        def mode(*args):
            self.arm.frames.append(('mode', args))
            self.arm.mode = args[1]
        self.arm.piper.MotionCtrl_2 = mode
        self.arm.piper.GetArmStatus = lambda: SimpleNamespace(
            time_stamp=self.clock[0], arm_status=self.arm.status())
        self.follower = MitFollower(self.leader, self.arm, guard=JointGuard(common_limits()), log=True)
        self.patches = [patch('mit_follower.time.time', side_effect=lambda: self.clock[0]),
                        patch('mit_follower.time.monotonic', side_effect=lambda: self.clock[0]),
                        patch('mit_follower.time.sleep', side_effect=self.sleep)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def sleep(self, dt):
        self.clock[0] += dt

    def test_entry_seeds_actual_pose_and_waits_for_mode(self):
        self.follower.start()
        self.assertGreaterEqual(self.clock[0], 1000.05)
        self.assertIn(('mode', (1,6,0,0xAD,0,0)), self.arm.frames)
        frames = [f[1] for f in self.arm.frames if f[0]=='mit']
        for j, pos, vel, kp, kd, torque in frames:
            self.assertEqual(pos, self.arm.pose[j-1])
            self.assertEqual((vel,kp,kd,torque), (0.,10.,.8,0.))

    def test_live_gains_are_ramped_and_no_position_frames_during_tracking(self):
        self.follower.start()
        self.arm.frames.clear()
        self.follower.set_gain('follower_kp', 20)
        self.follower.set_gain('follower_kd', 1.5)
        self.assertEqual(self.arm.frames, [])
        self.leader.pose[0] = .2
        self.sleep(.01)
        self.follower.tick()
        self.assertEqual(len(self.arm.frames), 6)
        self.assertTrue(all(f[0]=='mit' for f in self.arm.frames))
        self.assertAlmostEqual(self.arm.frames[0][1][1], .005)
        self.assertAlmostEqual(self.arm.frames[0][1][3], 10.05)
        self.assertAlmostEqual(self.arm.frames[0][1][4], .805)

    def test_tracking_error_mode_loss_stale_and_deadline_reject_before_send(self):
        for failure in ('error','mode','stale','deadline'):
            self.arm.pose = np.array([0,1,-1,0,0,0], float)
            self.arm.obs_time = lambda: self.clock[0]
            self.follower.start()
            self.arm.frames.clear()
            self.sleep(.01)
            if failure == 'error': self.arm.pose[0] += .1
            if failure == 'mode': self.arm.mode = 1
            if failure == 'stale': self.arm.obs_time = lambda: 1
            if failure == 'deadline': self.sleep(.2)
            with self.assertRaises(RuntimeError): self.follower.tick()
            self.assertEqual(self.arm.frames, [])

    def test_stop_switches_to_position_hold_not_mit(self):
        self.follower.start()
        self.arm.frames.clear()
        self.follower.stop()
        self.assertEqual(self.arm.frames[0], ('mode', (1,1,20,0,0,0)))
        self.assertEqual([f[0] for f in self.arm.frames[1:]], ['joint']*3)

    def test_gain_validation_and_panel_routing(self):
        for name, value in [('follower_kp',0), ('follower_kp',31),
                            ('follower_kd',0), ('follower_kd',float('nan'))]:
            with self.assertRaises(ValueError): self.follower.set_gain(name,value)
        with self.assertRaises(ValueError): self.follower.set_speed('follower_speed_pct',30)
        law = ReferenceLaw(None, BalancedDrag(DynamicsAdapter(FakeDynamics())), [0]*6, [1]*6)
        law.follower = self.follower
        law.set('follower_kp',15)
        self.assertEqual(law.state()['follower_kp'],15)
        self.assertIsNone(law.state()['follower_speed_pct'])

    def test_entry_timeout_does_not_report_success(self):
        self.arm.piper.GetArmStatus = lambda: SimpleNamespace(
            time_stamp=1., arm_status=SimpleNamespace(mode_feed=1))
        with self.assertRaisesRegex(RuntimeError,'failed to enter MIT'):
            self.follower.start()
        self.follower.stop()
        self.assertEqual(self.arm.frames[-1][0], 'joint')


if __name__ == '__main__':
    unittest.main()
