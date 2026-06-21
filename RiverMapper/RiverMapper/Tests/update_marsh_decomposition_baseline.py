"""Deliberately regenerate the approved marsh decomposition baseline.

Run this only after inspecting and approving an intentional geometry change::

    python -m RiverMapper.Tests.update_marsh_decomposition_baseline
"""

import json

from shapely import wkt

from RiverMapper import marsh
from .test_marsh_decomposition import (
    BASELINE_PATH,
    LAYER_NAMES,
    layer_summary,
)


def main():
    fixture = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    original = wkt.loads(fixture["input_wkt"])

    config = marsh.make_config("standard", fixture["parameters"])
    result = marsh.decompose_marsh_polygon(original, config)

    fixture["expected"] = {
        name: layer_summary(geometries)
        for name, geometries in zip(LAYER_NAMES, result)
    }
    BASELINE_PATH.write_text(
        json.dumps(fixture, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Updated: {BASELINE_PATH}")
    print("Review the JSON diff and resulting geometries before committing.")


if __name__ == "__main__":
    main()
