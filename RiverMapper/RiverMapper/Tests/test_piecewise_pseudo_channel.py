"""Piecewise pseudo banks preserve smooth stretches and scale with channel width."""
import unittest

import numpy as np

from RiverMapper.config_river_map import ConfigRiverMap
from RiverMapper.make_river_map import blend_pseudo_channel


class TestPiecewisePseudoChannel(unittest.TestCase):
    def setUp(self):
        self.line = np.c_[np.arange(0, 602, 2), np.zeros(301), np.zeros(301)]
        self.x = self.line[:, 0].copy()
        self.left = np.full(301, 10.0)
        self.right = -self.left

    def blend(self, line=None, scale=1.0):
        return blend_pseudo_channel(
            self.line*scale if line is None else line,
            self.x*scale, self.left*scale, self.x*scale, self.right*scale, 5*scale)

    def test_smooth_banks_are_unchanged(self):
        xl, yl, xr, yr, weight = self.blend()
        np.testing.assert_array_equal(weight, 0)
        for actual, expected in zip([xl, yl, xr, yr], [self.x, self.left, self.x, self.right]):
            np.testing.assert_array_equal(actual, expected)

    def test_only_sustained_rough_stretch_is_blended(self):
        self.left[100:201] += 5 * (-1.0)**np.arange(101)
        xl, yl, xr, yr, weight = self.blend()
        self.assertTrue(np.any(weight == 1))
        self.assertTrue(np.any((weight > 0) & (weight < 1)))
        self.assertEqual(weight[0], 0)
        self.assertEqual(weight[-1], 0)
        for actual, expected in zip([xl, yl, xr, yr], [self.x, self.left, self.x, self.right]):
            np.testing.assert_array_equal(actual[weight == 0], expected[weight == 0])
        np.testing.assert_allclose(np.hypot(xl-xr, yl-yr)[weight == 1], 5, atol=1e-10)

    def test_selection_is_invariant_to_uniform_length_scaling(self):
        self.left[100:201] += 5 * (-1.0)**np.arange(101)
        original = self.blend()
        scaled = self.blend(scale=3)
        np.testing.assert_allclose(scaled[-1], original[-1], atol=1e-12)
        for actual, expected in zip(scaled[:-1], original[:-1]):
            np.testing.assert_allclose(actual, expected*3, atol=1e-10)

    def test_isolated_bank_spike_is_not_a_sustained_stretch(self):
        self.left[150] += 10
        np.testing.assert_array_equal(self.blend()[-1], 0)

    def test_matching_thalweg_and_bank_turns_do_not_trigger_fallback(self):
        self.line[:, 1] = 5*np.sin(self.x/10)
        self.left = self.line[:, 1]+10
        self.right = self.line[:, 1]-10
        np.testing.assert_array_equal(self.blend()[-1], 0)

    def test_short_lines_and_invalid_ratios(self):
        result = blend_pseudo_channel(self.line[:2], self.x[:2], self.left[:2],
                                      self.x[:2], self.right[:2], 5)
        np.testing.assert_array_equal(result[-1], 0)
        for ratios in [(0, 3, 1), (5, 3), (5, np.nan, 1)]:
            with self.assertRaises(ValueError):
                blend_pseudo_channel(self.line, self.x, self.left, self.x, self.right, 5, ratios)

    def test_config_forwards_options_and_defaults_to_disabled(self):
        self.assertFalse(ConfigRiverMap().optional['i_piecewise_pseudo_channel'])
        config = ConfigRiverMap(i_piecewise_pseudo_channel=True, piecewise_pseudo_channel_ratios=(6, 4, 2))
        self.assertTrue(config.optional['i_piecewise_pseudo_channel'])
        self.assertEqual(config.optional['piecewise_pseudo_channel_ratios'], (6, 4, 2))


if __name__ == '__main__':
    unittest.main()
