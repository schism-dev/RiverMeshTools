"""Regression tests for the marsh fleshy/skinny decomposition.

Run from the repository root with the RiverMapper environment::

    python -m unittest RiverMapper.Tests.test_marsh_decomposition

The fingerprint assertions intentionally make geometry changes visible.  When
a reviewed algorithm change is correct, regenerate the values in the baseline
JSON deliberately; do not merely weaken or remove the assertions.
"""

from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

from shapely import wkt
from shapely.ops import unary_union


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "Scripts" / "marsh_decomposition.py"
BASELINE_PATH = Path(__file__).parent / "data" / "marsh_decomposition_baseline.json"

LAYER_NAMES = (
    "fleshy",
    "skinny",
    "core",
    "filtered",
    "fleshy_mask",
    "candidate_skinny_mask",
    "candidate_skinny_raw",
)


def load_script_module():
    """Load the script as a library without executing its main workflow."""
    spec = importlib.util.spec_from_file_location("marsh_decomposition", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@contextmanager
def configured_module(module, parameters):
    """Apply fixture parameters and restore the module globals afterward."""
    core_distance = parameters["fleshy_core_dist"]
    global_parameters = {
        key: value
        for key, value in parameters.items()
        if key != "fleshy_core_dist"
    }
    previous = {key: getattr(module, key) for key in global_parameters}
    try:
        for key, value in global_parameters.items():
            setattr(module, key, value)
        yield core_distance
    finally:
        for key, value in previous.items():
            setattr(module, key, value)


class TestMarshDecompositionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script_module()
        cls.fixture = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        cls.original = wkt.loads(cls.fixture["input_wkt"])

    def decompose(self):
        with configured_module(self.module, self.fixture["parameters"]) as core_distance:
            # Cleanup warnings are useful in production but just add noise to
            # successful regression-test output.
            with redirect_stdout(io.StringIO()):
                result = self.module.decompose_marsh_polygon(
                    self.original,
                    fleshy_core_dist=core_distance,
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
                    "Geometry changed. Review the output before approving a new baseline.",
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
            {name: geometry_fingerprint(geoms) for name, geoms in first.items()},
            {name: geometry_fingerprint(geoms) for name, geoms in second.items()},
        )


if __name__ == "__main__":
    unittest.main()
