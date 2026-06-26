import unittest

from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point

from RiverMapper.make_river_map import (
    line_geometries_from_union,
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


if __name__ == "__main__":
    unittest.main()
