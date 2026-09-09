import unittest
from types import SimpleNamespace

import numpy as np

from joint_breakaway import JointBreakawayDrag
from b601_reference_balance import BalancedDrag as Reference
from b601_drag import ReferenceLaw, DynamicsAdapter
from test_b601_reference import FakeDynamics


class JointBreakawayTests(unittest.TestCase):
    def make(self, cls=JointBreakawayDrag, **kwargs):
        c = cls(DynamicsAdapter(FakeDynamics()), kappa=0, fric_scale=.9,
                f_static=np.ones(6), r0=np.full(6,.1), **kwargs)
        c.set_kappa(0)
        c._observe = lambda *args: None
        c._runaway_check = lambda *args: None
        c._g_abs = np.zeros(6)
        c._M = np.eye(6)
        c.r = np.array([1,-1,1,-1,1,-1.])
        return c

    def test_independent_fractions_and_full_fraction_per_joint(self):
        c = self.make()
        for j in range(6):
            c.set_break(0)
            c.set_joint_break(j,1)
            out = c.update(np.zeros(6),np.zeros(6))
            expected = np.zeros(6)
            expected[j] = .9*min(1.,c.level_max)*np.sign(c.r[j])
            np.testing.assert_allclose(c.breakaway_torque,expected)
            np.testing.assert_allclose(out,np.clip(expected,-c.tau_cap,c.tau_cap))

    def test_uniform_matches_snapshot_with_nonzero_assist(self):
        a, b = self.make(break_beta=.7), self.make(Reference,break_beta=.7)
        for v in (np.zeros(6), np.full(6,.05),np.full(6,-.1)):
            np.testing.assert_allclose(a.update(np.zeros(6),v),b.update(np.zeros(6),v),atol=1e-12)

    def test_zero_friction_disables_assist_and_saturation_is_reported(self):
        c = self.make(break_beta=1, tau_cap=np.full(6,.01))
        c.update(np.zeros(6),np.zeros(6))
        self.assertTrue(c.saturated.all())
        c.set_fric_scale(0)
        c.update(np.zeros(6),np.zeros(6))
        np.testing.assert_array_equal(c.breakaway_torque,np.zeros(6))
        self.assertFalse(c.saturated.any())

    def test_panel_validation_and_logging(self):
        c = self.make()
        law = ReferenceLaw(SimpleNamespace(gravity_torque=lambda q:np.zeros(6)),c,[0]*6,[8]*6)
        law.log_enabled=True
        c.set_joint_break(1,.5)
        law.set('break_beta',1)
        self.assertEqual(law.state()['break_beta'],1)
        np.testing.assert_array_equal(c.break_fractions,np.ones(6))
        for key,value in [('break_beta_j2',1),('break_beta',1.1),('break_beta',float('nan'))]:
            with self.assertRaises(ValueError):law.set(key,value)
        law(SimpleNamespace(q=np.zeros(6),t=0))
        self.assertEqual(len(law.breakaway_rows[0]),24)
        self.assertEqual(law.breakaway_rows[0][1],1)


if __name__ == '__main__':
    unittest.main()
