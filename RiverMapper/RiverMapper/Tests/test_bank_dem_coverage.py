"""Bank searches use connected DEM coverage while thalwegs require complete coverage."""
import unittest
from unittest.mock import patch

import numpy as np

from RiverMapper.make_river_map import correct_thalwegs, get_bank
from RiverMapper.river_map_tif_preproc import DemData, get_elev_from_tiles
from RiverMapper.SMS import lonlat2cpp


class TestBankDemCoverage(unittest.TestCase):
    @staticmethod
    def tile(elevations):
        lon = np.arange(len(elevations)) * 1e-5
        lat = np.array([0.0, 1e-5])
        return DemData(lon, lat, lon, lat, np.tile(elevations, (2, 1)), 1e-5, 1e-5)

    @staticmethod
    def search(tiles, scale=1.0):
        lon = tiles[0].lon
        x, y = lonlat2cpp(np.array([lon[0]]), np.array([0.0]))
        xt, yt = lonlat2cpp(np.array([lon[-1]]), np.array([0.0]))
        return get_bank(tiles, x, y, np.array([0.0]), xt, yt, len(lon), scale)

    def assert_bank_at(self, result, tile, index, unresolved):
        expected_x, expected_y = lonlat2cpp(np.array([tile.lon[index]]), np.array([0.0]))
        np.testing.assert_allclose(result[0], expected_x)
        np.testing.assert_allclose(result[1], expected_y)
        np.testing.assert_array_equal(result[2], [unresolved])

    def test_bank_before_nodata_is_retained(self):
        tile = self.tile([-2.0, -1.0, 2.0, np.nan, np.nan])
        self.assert_bank_at(self.search([tile]), tile, 2, False)

    def test_search_does_not_jump_across_a_gap(self):
        tile = self.tile([-2.0, -1.0, np.nan, 2.0, 3.0])
        self.assert_bank_at(self.search([tile]), tile, 1, True)

    def test_raster_nodata_sentinel_truncates_search(self):
        tile = self.tile([-2.0, -1.0, 3.4e38, 2.0, 3.0])
        self.assert_bank_at(self.search([tile]), tile, 1, True)

    def test_missing_thalweg_elevation_is_rejected(self):
        tile = self.tile([np.nan, -1.0, 2.0, 3.0, 4.0])
        self.assertEqual(self.search([tile]), (None, None, None))

    def test_complete_coverage_preserves_bank_detection(self):
        tile = self.tile([-2.0, -1.0, 2.0, 3.0, 4.0])
        self.assert_bank_at(self.search([tile]), tile, 2, False)

    def test_closest_elevation_stays_before_gap(self):
        tile = self.tile([3.0, 1.0, np.nan, 0.1, 0.2])
        self.assert_bank_at(self.search([tile]), tile, 1, False)

    def test_secondary_tile_can_complete_the_search(self):
        first = self.tile([-2.0, -1.0, np.nan, np.nan, np.nan])
        second = self.tile([-4.0, -3.0, 2.0, 3.0, 4.0])
        self.assert_bank_at(self.search([first, second]), first, 2, False)

    def test_inverted_elevations_preserve_coverage_mask(self):
        tile = self.tile([2.0, 1.0, -2.0, np.nan, np.nan])
        self.assert_bank_at(self.search([tile], scale=-1.0), tile, 2, False)

    def test_sampler_default_still_requires_full_coverage(self):
        tile = self.tile([-2.0, -1.0, np.nan, 2.0, 3.0])
        x, y = lonlat2cpp(tile.lon, np.zeros(len(tile.lon)))
        self.assertIsNone(get_elev_from_tiles(x, y, [tile]))
        partial = get_elev_from_tiles(x, y, [tile], allow_missing=True)
        np.testing.assert_allclose(partial, tile.elev[0], equal_nan=True)


class TestThalwegCorrectionCoverage(unittest.TestCase):
    def setUp(self):
        self.line = np.array([[0, 0, 1], [10, 0, 2]])

    def correct(self, left, right, search_length=2.0):
        with patch('RiverMapper.make_river_map.get_elev_from_tiles', side_effect=[left, right]):
            return correct_thalwegs([], 1.0, self.line, search_length, np.zeros(2), '')

    def test_no_candidates_preserves_original_vertices(self):
        unavailable = np.full((2, 5), np.nan)
        unavailable[1, 0] = 1.0
        corrected, success = self.correct(unavailable, unavailable)
        np.testing.assert_array_equal(corrected, self.line)
        self.assertFalse(success)

    def test_each_vertex_uses_its_covered_side_without_crossing_gaps(self):
        left = np.array([[5, 4, 3, np.nan, -100], [8, np.nan, 0, 0, 0]])
        right = np.array([[5, np.nan, 0, 0, 0], [8, 7, 6, np.nan, -100]])
        corrected, success = self.correct(left, right)
        np.testing.assert_allclose(corrected[:, 0], [-0.5, 10.5])
        np.testing.assert_allclose(corrected[:, 1], self.line[:, 1], atol=1e-12)
        np.testing.assert_array_equal(corrected[:, 2], self.line[:, 2])
        self.assertTrue(success)

    def test_lowest_sample_count_is_capped_at_ten(self):
        left = np.tile(np.arange(12, 0, -1, dtype=float), (2, 1))
        right = np.full((2, 12), np.nan)
        right[:, 0] = 12
        corrected, success = self.correct(left, right, search_length=6.0)
        np.testing.assert_allclose(corrected[:, 0], self.line[:, 0] - 6 * 6 / 11)
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
