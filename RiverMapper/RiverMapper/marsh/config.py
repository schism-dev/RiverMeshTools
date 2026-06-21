"""Configuration, recipes, and command-line parsing for marsh decomposition."""

import argparse
from dataclasses import asdict, dataclass, field, fields, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class MarshConfig:
    """Reusable numerical parameters for marsh decomposition."""

    # Reference-geometry filtering and decomposition.
    use_filter_for_distance: bool = True
    filter_dist: float = 5.0
    filter_min_area: float = 0.0
    skinny_full_width_threshold: float = 20.0
    # None derives the core distance from half of the full-width threshold.
    fleshy_core_dist: float | None = None
    buffer_join_style: int = 1  # 1=round, 2=mitre, 3=bevel

    # Small-part cleanup.
    small_skinny_area_ratio: float = 0.001
    small_fleshy_area_ratio: float = 0.01
    small_polygon_cleanup_mode: str = "direct_discard"
    max_cleanup_iter: int = 3
    discard_remaining_small_parts: bool = True

    # RiverMapper design and paving parameters.
    X2: float = 10.0
    Y: float = 10.0
    Z: float = 30.0
    fleshy_resolution_threshold: float = 30.0
    small_fleshy_resolution_factor: float = 0.4

    # Skeleton extraction and diagnostics.
    skeleton_dx: float = 0.2
    skeleton_vertex_spacing: float = 1.0
    skeleton_random_seed: int = 0
    min_skeleton_line_length: float = 0.0
    diagnostic_min_area: float = 0.0

    @property
    def effective_fleshy_core_dist(self):
        if self.fleshy_core_dist is not None:
            return self.fleshy_core_dist
        return 0.5 * self.skinny_full_width_threshold

    def validate(self):
        positive = {
            "skinny_full_width_threshold": self.skinny_full_width_threshold,
            "fleshy_core_dist": self.effective_fleshy_core_dist,
            "X2": self.X2,
            "Y": self.Y,
            "Z": self.Z,
            "skeleton_dx": self.skeleton_dx,
            "skeleton_vertex_spacing": self.skeleton_vertex_spacing,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")

        nonnegative = {
            "filter_dist": self.filter_dist,
            "filter_min_area": self.filter_min_area,
            "small_skinny_area_ratio": self.small_skinny_area_ratio,
            "small_fleshy_area_ratio": self.small_fleshy_area_ratio,
            "min_skeleton_line_length": self.min_skeleton_line_length,
            "diagnostic_min_area": self.diagnostic_min_area,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative; got {value}")

        cleanup_modes = {"direct_discard", "iterative"}
        if self.small_polygon_cleanup_mode not in cleanup_modes:
            raise ValueError(
                "small_polygon_cleanup_mode must be 'direct_discard' "
                "or 'iterative'"
            )
        if self.buffer_join_style not in {1, 2, 3}:
            raise ValueError("buffer_join_style must be 1, 2, or 3")
        if self.max_cleanup_iter < 1:
            raise ValueError("max_cleanup_iter must be at least 1")
        if self.skeleton_random_seed < 0:
            raise ValueError("skeleton_random_seed must be nonnegative")


@dataclass(frozen=True)
class MarshRunConfig:
    """Files and numerical parameters resolved for one execution."""

    input_file: Path = Path(
        "/sciclone/schism10/feiye/Marsh/Shapefiles/marsh_test1.shp"
    )
    output_file: Path = Path(
        "/sciclone/schism10/feiye/Marsh/Shapefiles/"
        "marsh_test1_decomposed.gpkg"
    )
    parameters: MarshConfig = field(default_factory=MarshConfig)


# Each recipe contains only values that differ from the standard, historically
# tested configuration. Add project-specific recipes here or save them in JSON.
RECIPE_OVERRIDES = {
    "standard": {},
    "fast_preview": {
        "skeleton_dx": 1.0,
        "skeleton_vertex_spacing": 5.0,
        "min_skeleton_line_length": 2.0,
    },
    "high_detail": {
        "skeleton_dx": 0.1,
        "skeleton_vertex_spacing": 0.5,
    },
    "coverage_preserving": {
        "small_polygon_cleanup_mode": "iterative",
        "discard_remaining_small_parts": False,
    },
}

RECIPE_DESCRIPTIONS = {
    "standard": "Historically tested production settings.",
    "fast_preview": "Coarser skeleton for quick parameter exploration.",
    "high_detail": "Finer skeleton for final high-resolution output.",
    "coverage_preserving": (
        "Iterative cleanup without deliberate discarded gaps."
    ),
}


def config_to_dict(config):
    """Return a JSON-serializable configuration dictionary."""
    return asdict(config)


def make_config(recipe="standard", values=None):
    """Build and validate a configuration from a recipe plus overrides."""
    if recipe not in RECIPE_OVERRIDES:
        choices = ", ".join(sorted(RECIPE_OVERRIDES))
        raise ValueError(
            f"Unknown recipe {recipe!r}. Available recipes: {choices}"
        )

    updates = dict(RECIPE_OVERRIDES[recipe])
    updates.update(values or {})
    valid_names = {field.name for field in fields(MarshConfig)}
    unknown = sorted(set(updates) - valid_names)
    if unknown:
        raise ValueError(
            f"Unknown configuration parameter(s): {', '.join(unknown)}"
        )

    config = replace(MarshConfig(), **updates)
    config.validate()
    return config


def parse_override(text):
    """Parse a command-line NAME=VALUE override using JSON scalar syntax."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            "overrides must have the form NAME=VALUE"
        )
    name, raw_value = text.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError(
            "override parameter name cannot be empty"
        )
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return name, value


def load_config_file(filename):
    """Load a current or legacy JSON recipe document."""
    document = json.loads(Path(filename).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Configuration JSON must contain an object")

    if "parameters" in document:
        parameters = document["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("The JSON 'parameters' value must be an object")
        extra = set(document) - {
            "recipe",
            "description",
            "input_file",
            "output_file",
            "parameters",
        }
        if extra:
            raise ValueError(
                "Unknown top-level configuration key(s): "
                f"{', '.join(sorted(extra))}"
            )
    else:
        parameters = dict(document)
        parameters.pop("description", None)
        parameters.pop("recipe", None)

    # Older saved recipes placed paths inside the parameters object.
    input_file = document.get("input_file", parameters.pop("input_file", None))
    output_file = document.get(
        "output_file",
        parameters.pop("output_file", None),
    )

    return document.get("recipe"), input_file, output_file, parameters


def parse_run_config(argv=None):
    """Resolve recipe, JSON, and CLI parameters in precedence order."""
    parser = argparse.ArgumentParser(
        description="Decompose marsh polygons into fleshy and skinny regions."
    )
    parser.add_argument("--recipe", choices=sorted(RECIPE_OVERRIDES))
    parser.add_argument(
        "--config", type=Path, help="JSON file with saved parameters"
    )
    parser.add_argument("--input", dest="input_file", type=Path)
    parser.add_argument("--output", dest="output_file", type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        type=parse_override,
        metavar="NAME=VALUE",
        help="override any parameter; repeat this option for multiple values",
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="print built-in recipes and exit",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="print the resolved setup before running",
    )
    parser.add_argument(
        "--write-config",
        type=Path,
        metavar="FILE",
        help="save reusable recipe parameters as JSON and exit",
    )
    args = parser.parse_args(argv)

    if args.list_recipes:
        print("Built-in marsh decomposition recipes:")
        for name, overrides in RECIPE_OVERRIDES.items():
            print(f"  {name}: {RECIPE_DESCRIPTIONS[name]}")
            print(f"    overrides: {json.dumps(overrides, sort_keys=True)}")
        return None

    file_recipe = None
    file_input = None
    file_output = None
    file_values = {}
    if args.config is not None:
        file_recipe, file_input, file_output, file_values = load_config_file(
            args.config
        )

    recipe = args.recipe or file_recipe or "standard"
    values = dict(file_values)
    values.update(dict(args.overrides))

    config = make_config(recipe, values)
    serialized = config_to_dict(config)
    defaults = MarshRunConfig()
    run_config = MarshRunConfig(
        input_file=args.input_file or file_input or defaults.input_file,
        output_file=args.output_file or file_output or defaults.output_file,
        parameters=config,
    )

    run_document = {
        "recipe": recipe,
        "input_file": str(run_config.input_file),
        "output_file": str(run_config.output_file),
        "parameters": serialized,
    }

    if args.show_config:
        shown = dict(run_document)
        shown["parameters"] = dict(serialized)
        shown["parameters"]["effective_fleshy_core_dist"] = (
            config.effective_fleshy_core_dist
        )
        print(json.dumps(shown, indent=2))

    if args.write_config is not None:
        recipe_document = {
            "recipe": recipe,
            "parameters": serialized,
        }
        args.write_config.write_text(
            json.dumps(recipe_document, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved configuration: {args.write_config}")
        return None

    return recipe, run_config
