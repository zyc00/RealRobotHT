import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from b601_drag import BalancedDrag, DynamicsAdapter, ReferenceLaw, VelocityEstimator


class FakeDynamics:
    nq = 6
    def mass_matrix(self, q):
        return np.diag([.15, 1.1, .32, .03, .006, .01])
    def coriolis(self, q, v):
        return np.zeros((6, 6))
    def gravity(self, q):
        return np.array([.18, -4, -2, .1, .1, 0])
    def jacobian(self, q, frame):
        return np.diag([.5, .6, .4, 1, 1, 1])


class ReferenceTests(unittest.TestCase):
    def test_core_is_unmodified_snapshot(self):
        path = Path(__file__).with_name('b601_reference_balance.py')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '888329aa0fee08c9b062dc5f895763397337e86d001f779065f6ef4dbbeac471')

    def test_tool_axes_preserve_virtual_joint_inertia(self):
        dyn = FakeDynamics(); adapter = DynamicsAdapter(dyn)
        j = dyn.jacobian(None, 'local'); mapped = adapter.ee_jacobian(None)
        piper_inertia = np.diag([1.8]*3 + [.06, .06, .0005])
        reference_inertia = np.diag([1.8]*3 + [.0005, .06, .06])
        np.testing.assert_allclose(j.T @ piper_inertia @ j, mapped.T @ reference_inertia @ mapped)

    def test_velocity_filter(self):
        estimator = VelocityEstimator()
        np.testing.assert_array_equal(estimator.update(np.zeros(6), .01), np.zeros(6))
        np.testing.assert_allclose(estimator.update(np.full(6, .01), .01), .25)
        np.testing.assert_allclose(estimator.update(np.full(6, .02), .01), .4375)

    def test_original_b601_replay_matches_piper_adapter(self):
        path = Path('/home/yuchen/projects/b601_teleop/b601/balance.py')
        if not path.exists():
            self.skipTest('original B601 checkout unavailable; snapshot hash test still applies')
        spec = importlib.util.spec_from_file_location('upstream_reference', path)
        upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
        dyn = DynamicsAdapter(FakeDynamics())
        config = dict(kappa=2, fric_scale=.8, damp_t=2, damp_r=.3, break_beta=.2,
                      gate_floor=.5, alpha_sigma_v=.03, tau_cap=np.full(6, .5))
        core = BalancedDrag(dyn, **config)
        original = upstream.BalancedDrag(dyn, **config)
        law = ReferenceLaw(SimpleNamespace(gravity_torque=dyn.gravity), core,
                           [.1]*6, [.1]*6)
        law.log_enabled = True
        velocity = VelocityEstimator()
        # Standstill (bias learning), movement, reversals, clipping, live toggles.
        previous_t = None
        for i in range(650):
            t = i*.01
            q = np.zeros(6) if i < 150 else .03*np.sin(t*np.arange(1,7))
            if i in (300, 400):
                k = 0 if i == 300 else 1.5
                core.set_kappa(k); original.set_kappa(k)
            dt = .01 if previous_t is None else t-previous_t
            previous_t = t
            v = velocity.update(q, dt)
            expected = original.update(q, v, dt)
            ff = np.clip(dyn.gravity(q)+expected, -.1, .1)
            original.note_sent(ff, np.zeros(6), law.kd, q, np.zeros(6))
            cmd = law(SimpleNamespace(q=q, qdot=np.full(6, 999), t=t))
            np.testing.assert_allclose(cmd.t_ff, ff, atol=1e-12)
            np.testing.assert_allclose(core.r, original.r, atol=1e-12)
            np.testing.assert_allclose(core.tau_out, original.tau_out, atol=1e-12)
            np.testing.assert_allclose(core._sent[0], cmd.t_ff)
            self.assertEqual(core.trips, original.trips)
        self.assertEqual(len(law.rows), 650)

    def test_panel_rejects_bad_inputs(self):
        dyn = DynamicsAdapter(FakeDynamics())
        law = ReferenceLaw(None, BalancedDrag(dyn), [0]*6, [1]*6)
        for name, value in [('kappa', 3), ('fric_scale', float('nan')), ('break_beta', -1)]:
            with self.assertRaises(ValueError):
                law.set(name, value)
        law.set('kappa', 0)
        self.assertFalse(law.core.shaping_on)


if __name__ == '__main__':
    unittest.main()
