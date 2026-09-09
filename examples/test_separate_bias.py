"""Synthetic validation of pairing and independent bias estimation."""
import unittest

import numpy as np

from separate_bias import collect, fit


def row(sign, midpoint=0.2):
    r = dict(kind='kin', vset=str(sign * 0.1))
    for j in range(1, 7):
        r.update({f'q{j}': '0', f'mv{j}': str(sign * 0.1),
                  f'f{j}': str(midpoint + sign * 0.5)})
    return r


class BiasTests(unittest.TestCase):
    def test_pairing_preserves_repeats_and_rejects_nonmotion(self):
        p, m = row(1), row(-1)
        m['mv2'] = '0'
        m['mv3'] = '-0.01'
        m['f4'] = 'nan'
        _, pairs, unmatched = collect([p, m, row(1, 0.3), row(-1, 0.3), row(1)])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(unmatched, 1)
        self.assertAlmostEqual(pairs[0][1][0], 0.2)
        self.assertAlmostEqual(pairs[0][2][0], 0.5)
        self.assertTrue(np.isnan(pairs[0][1][1:4]).all())
        np.testing.assert_allclose(pairs[1][1], 0.3)

    def test_separate_known_scale_and_bias(self):
        q = np.arange(36).reshape(6, 6) * 0.01
        gravity = np.tile(np.arange(6)[:, None], (1, 6))
        static = fit(q, gravity * 0.1 + 0.3, gravity)
        moving = fit(q, gravity * 0.05 - 0.2, gravity)
        np.testing.assert_allclose(static['scale'], 1.1)
        np.testing.assert_allclose(static['bias'], 0.3)
        np.testing.assert_allclose(moving['scale'], 1.05)
        np.testing.assert_allclose(moving['bias'], -0.2)

    def test_equal_pose_weight_and_missing_support(self):
        q = np.zeros((12, 6))
        q[10] = 1
        q[11] = 2
        r = np.zeros((12, 6))
        r[10:] = 0.6
        r[10:, 1] = np.nan
        result = fit(q, r, np.zeros_like(q))
        self.assertAlmostEqual(result['bias'][0], 0.4)
        self.assertEqual(result['poses'][0], 3)
        self.assertEqual(result['n'][0], 12)
        self.assertTrue(np.isnan(result['bias'][1]))
        self.assertEqual(result['poses'][1], 1)


if __name__ == '__main__':
    unittest.main()
