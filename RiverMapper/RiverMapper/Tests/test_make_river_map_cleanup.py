import unittest

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point

from RiverMapper.make_river_map import (
    line_geometries_from_union,
    point_to_candidate_segment_distances,
    snap_closeby_points_global,
    union_line_geometries,
)


class TestMakeRiverMapCleanup(unittest.TestCase):
    def test_line_geometries_from_union_flattens_line_parts(self):
        line_a = LineString([(0, 0), (1, 0)])
        line_b = LineString([(1, 0), (2, 0)])
        line_c = LineString([(0, 1), (1, 1)])

        geometries = line_geometries_from_union(
            GeometryCollection(
                [
                    line_a,
                    MultiLineString([line_b, line_c]),
                    Point(5, 5),
                ]
            )
        )

        self.assertEqual(geometries, [line_a, line_b, line_c])

    def test_union_line_geometries_nodes_intersections_without_geopandas(self):
        arcs = [
            LineString([(0, 0), (2, 0)]),
            LineString([(1, -1), (1, 1)]),
        ]

        noded_arcs = union_line_geometries(arcs)

        self.assertEqual(len(noded_arcs), 4)
        self.assertTrue(
            all(isinstance(arc, LineString) for arc in noded_arcs)
        )

    def test_candidate_segment_distances_are_vectorized_per_point(self):
        points = np.array([[1.0, 1.0], [4.0, 0.0]])
        candidate_segments = np.array(
            [
                [
                    [0.0, 0.0, 2.0, 0.0],
                    [3.0, 3.0, 3.0, 5.0],
                ],
                [
                    [0.0, 0.0, 2.0, 0.0],
                    [3.0, -1.0, 3.0, 1.0],
                ],
            ]
        )

        distances = point_to_candidate_segment_distances(
            points,
            candidate_segments,
        )

        np.testing.assert_allclose(
            distances,
            np.array(
                [
                    [1.0, np.sqrt(8.0)],
                    [2.0, 1.0],
                ]
            ),
        )

    def test_point_snapping_keeps_local_width_thresholds(self):
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 1.0],
                [10.0, 0.0, 10.0],
                [10.5, 0.0, 10.0],
            ]
        )

        snapped, _ = snap_closeby_points_global(
            points,
            snap_point_reso_ratio=0.1,
        )

        self.assertFalse(np.array_equal(snapped[0], snapped[1]))
        np.testing.assert_array_equal(snapped[2], snapped[3])


if __name__ == "__main__":
    unittest.main()
