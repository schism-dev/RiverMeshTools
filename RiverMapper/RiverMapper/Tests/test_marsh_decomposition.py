"""Regression tests for the marsh fleshy/skinny decomposition.

Run from the repository root with the RiverMapper environment::

    python -m unittest RiverMapper.Tests.test_marsh_decomposition

The fingerprint assertions intentionally make geometry changes visible.  When
a reviewed algorithm change is correct, regenerate the values in the baseline
JSON deliberately; do not merely weaken or remove the assertions.
"""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import unittest

from shapely import wkt
from shapely.geometry import LineString
from shapely.ops import unary_union

from RiverMapper import marsh
from RiverMapper.marsh.output import ARC_LAYER_SPECS, make_arc_line_records
from RiverMapper.marsh.workflow import (
    average_skeleton_width,
    filter_final_fleshy_parts,
)


BASELINE_PATH = (
    Path(__file__).parent / "data" / "marsh_decomposition_baseline.json"
)

LAYER_NAMES = (
    "fleshy",
    "skinny",
    "core",
    "filtered",
    "fleshy_mask",
    "candidate_skinny_mask",
    "candidate_skinny_raw",
)


def geometry_fingerprint(geometries):
    """Hash normalized, order-independent WKB for a geometry collection."""
    normalized_wkb = sorted(geom.normalize().wkb for geom in geometries)
    digest = hashlib.sha256()
    for blob in normalized_wkb:
        digest.update(len(blob).to_bytes(8, byteorder="big"))
        digest.update(blob)
    return digest.hexdigest()


def layer_summary(geometries):
    """Return the values stored in the approved baseline manifest."""
    return {
        "count": len(geometries),
        "area_m2": round(sum(geom.area for geom in geometries), 9),
        "sha256": geometry_fingerprint(geometries),
    }


class TestMarshDecompositionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = marsh
        cls.fixture = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        cls.original = wkt.loads(cls.fixture["input_wkt"])
        cls.config = cls.module.make_config(
            "standard",
            cls.fixture["parameters"],
        )

    def decompose(self):
        # Cleanup warnings are useful in production but just add noise to
        # successful regression-test output.
        with redirect_stdout(io.StringIO()):
            result = self.module.decompose_marsh_polygon(
                self.original,
                self.config,
            )
        return dict(zip(LAYER_NAMES, result))

    def test_geometry_matches_approved_baseline(self):
        """Counts, areas, and exact normalized geometry must be reviewed."""
        layers = self.decompose()

        for layer_name, geometries in layers.items():
            with self.subTest(layer=layer_name):
                expected = self.fixture["expected"][layer_name]
                self.assertEqual(len(geometries), expected["count"])
                self.assertAlmostEqual(
                    sum(geom.area for geom in geometries),
                    expected["area_m2"],
                    places=8,
                )
                self.assertEqual(
                    geometry_fingerprint(geometries),
                    expected["sha256"],
                    "Geometry changed; review it before approving a baseline.",
                )

    def test_final_classes_are_valid_disjoint_subsets(self):
        """Check invariants independently of the stored fingerprint."""
        layers = self.decompose()
        fleshy = unary_union(layers["fleshy"])
        skinny = unary_union(layers["skinny"])

        self.assertTrue(fleshy.is_valid)
        self.assertTrue(skinny.is_valid)
        self.assertLess(fleshy.intersection(skinny).area, 1.0e-9)
        self.assertLess(fleshy.difference(self.original).area, 1.0e-9)
        self.assertLess(skinny.difference(self.original).area, 1.0e-9)

        # direct_discard may leave tiny gaps, but must never create overlap or
        # geometry outside the original marsh polygon.
        classified = unary_union([fleshy, skinny])
        self.assertLess(self.original.difference(classified).area, 0.01)

    def test_repeated_calls_are_identical(self):
        """Catch accidental statefulness or nondeterminism within a run."""
        first = self.decompose()
        second = self.decompose()
        self.assertEqual(
            {
                name: geometry_fingerprint(geoms)
                for name, geoms in first.items()
            },
            {
                name: geometry_fingerprint(geoms)
                for name, geoms in second.items()
            },
        )

    def test_any_discard_produces_warning(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.module.decompose_marsh_polygon(self.original, self.config)

        self.assertIn("direct_discard excluded", output.getvalue())

    def test_arc_line_classification_matches_rivermapper_schema(self):
        classifications = {
            spec["layer"]: (spec["arc_pos"], spec["dummy"])
            for spec in ARC_LAYER_SPECS
        }

        self.assertEqual(
            classifications,
            {
                "fleshy_boundary_lines": ("regular", 0),
                "skinny_boundary_lines": ("left half", 0),
                "skinny_skeleton_lines": ("dummy", 1),
            },
        )

    def test_arc_line_records_can_be_built_from_memory(self):
        records = {
            "fleshy_boundary_lines": [
                {"parent_id": 0, "geometry": LineString([(0, 0), (4, 0)])}
            ],
            "skinny_boundary_lines": [
                {"parent_id": 0, "geometry": LineString([(0, 1), (4, 1)])}
            ],
            "skeleton_lines": [
                {"parent_id": 0, "geometry": LineString([(0, 2), (4, 2)])}
            ],
        }

        arc_records = make_arc_line_records(records, self.config)
        classes = {
            record["src_layer"]: (record["arc_pos"], record["dummy"])
            for record in arc_records
        }

        self.assertEqual(
            classes,
            {
                "fleshy_boundary_lines": ("regular", 0),
                "skinny_boundary_lines": ("left half", 0),
                "skinny_skeleton_lines": ("dummy", 1),
            },
        )
        self.assertTrue(all(r["resampled"] == "T" for r in arc_records))

    def test_recipe_can_be_overridden_without_mutating_it(self):
        config = self.module.make_config(
            "fast_preview",
            {"skinny_full_width_threshold": 30.0, "filter_dist": 2.5},
        )

        self.assertEqual(config.skeleton_dx, 1.0)
        self.assertEqual(config.filter_dist, 2.5)
        self.assertEqual(config.effective_fleshy_core_dist, 15.0)
        self.assertEqual(
            self.module.make_config("fast_preview").filter_dist,
            5.0,
        )

    def test_legacy_xyz_parameter_names_are_supported(self):
        config = self.module.make_config(
            "standard",
            {"X2": 12.0, "Y": 14.0, "Z": 40.0},
        )

        self.assertEqual(config.boundary_buffer_distance, 12.0)
        self.assertEqual(config.along_boundary_resolution, 12.0)
        self.assertEqual(config.skinny_centerline_spacing, 14.0)
        self.assertEqual(config.default_fleshy_paving_resolution, 40.0)

    def test_run_files_are_separate_from_recipe_parameters(self):
        self.assertFalse(hasattr(self.config, "input_file"))
        self.assertFalse(hasattr(self.config, "output_file"))

        recipe, run_config = self.module.parse_run_config(
            [
                "--recipe",
                "fast_preview",
                "--input",
                "example_input.shp",
                "--output",
                "example_output.gpkg",
            ]
        )

        self.assertEqual(recipe, "fast_preview")
        self.assertEqual(run_config.input_file, Path("example_input.shp"))
        self.assertEqual(run_config.output_file, Path("example_output.gpkg"))
        self.assertEqual(run_config.parameters.skeleton_dx, 1.0)

    def test_invalid_recipe_parameter_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown configuration parameter",
        ):
            self.module.make_config("standard", {"skeleton_resolution": 1.0})

    def test_optional_final_product_filters_drop_small_fleshy_outputs(self):
        layers = self.decompose()
        config = self.module.make_config(
            "standard",
            {"min_fleshy_area_m2": 1.0e6},
        )

        fleshy, dropped_fleshy = (
            filter_final_fleshy_parts(layers["fleshy"], config)
        )

        self.assertEqual(fleshy, [])
        self.assertEqual(len(dropped_fleshy), len(layers["fleshy"]))

    def test_fleshy_final_product_filter_is_disabled_by_default(self):
        layers = self.decompose()

        fleshy, dropped_fleshy = (
            filter_final_fleshy_parts(layers["fleshy"], self.config)
        )

        self.assertEqual(fleshy, layers["fleshy"])
        self.assertEqual(dropped_fleshy, [])

    def test_average_skeleton_width_uses_length_weighted_full_width(self):
        width = average_skeleton_width(
            [
                {"D_mean": 2.0, "length_m": 10.0},
                {"D_mean": 4.0, "length_m": 30.0},
            ],
            fallback_width=99.0,
        )

        self.assertAlmostEqual(width, 7.0)

    def test_average_skeleton_width_falls_back_without_skeleton_lines(self):
        self.assertEqual(average_skeleton_width([], fallback_width=5.0), 5.0)


if __name__ == "__main__":
    unittest.main()
