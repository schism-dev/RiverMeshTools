"""Core fleshy/skinny marsh decomposition algorithms."""

from dataclasses import dataclass

import numpy as np
from shapely.ops import unary_union

from .geometry import clean_geom, explode_to_polygons, filter_small_polygons


@dataclass(frozen=True)
class DecompositionResult:
    """Named geometry layers returned by marsh decomposition."""

    fleshy: list
    skinny: list
    core: list
    filtered: list
    fleshy_mask: list
    candidate_skinny_mask: list
    candidate_skinny_raw: list

    def __iter__(self):
        yield self.fleshy
        yield self.skinny
        yield self.core
        yield self.filtered
        yield self.fleshy_mask
        yield self.candidate_skinny_mask
        yield self.candidate_skinny_raw


def build_filtered_polygon_for_distance(poly, config):
    """Build the geometry used only for calculating the fleshy/skinny mask."""
    poly = clean_geom(poly)
    if poly is None:
        return None

    if not config.use_filter_for_distance or config.filter_dist <= 0.0:
        return poly

    filtered = poly.buffer(
        config.filter_dist,
        join_style=config.buffer_join_style,
    ).buffer(
        -config.filter_dist,
        join_style=config.buffer_join_style,
    )

    filtered = clean_geom(filtered)

    if filtered is None or filtered.is_empty:
        return poly

    return filtered


def cleanup_direct_discard(
    original_poly,
    candidate_skinny_raw_parts,
    config,
):
    """Directly discard small fleshy/skinny polygons.

    This does not preserve exact original coverage when pieces are discarded.
    It is useful when tiny slivers are clearly numerical artifacts.
    """
    original_poly = clean_geom(original_poly)
    if original_poly is None:
        return None, None

    original_area = original_poly.area

    small_skinny_area_threshold = (
        config.small_skinny_area_ratio * original_area
    )
    small_fleshy_area_threshold = (
        config.small_fleshy_area_ratio * original_area
    )

    large_skinny_parts = [
        p for p in candidate_skinny_raw_parts
        if p.area >= small_skinny_area_threshold
    ]

    if len(large_skinny_parts) == 0:
        skinny = None
        fleshy = original_poly
    else:
        skinny = clean_geom(unary_union(large_skinny_parts))
        fleshy = clean_geom(original_poly.difference(skinny))

    fleshy_parts = explode_to_polygons(fleshy)
    large_fleshy_parts = [
        p for p in fleshy_parts
        if p.area >= small_fleshy_area_threshold
    ]

    small_fleshy_parts = [
        p for p in fleshy_parts
        if p.area < small_fleshy_area_threshold
    ]

    discarded_area = sum(p.area for p in small_fleshy_parts)
    if discarded_area > 0:
        print(
            "Warning: direct_discard excluded "
            f"{len(small_fleshy_parts)} small fleshy polygons from both "
            "final classes; "
            f"discarded_area={discarded_area:.6f} m^2 "
            f"({discarded_area / original_area:.6%} of parent polygon)"
        )

    fleshy = (
        clean_geom(unary_union(large_fleshy_parts))
        if large_fleshy_parts
        else None
    )

    skinny_parts = explode_to_polygons(skinny)
    large_skinny_parts = [
        p for p in skinny_parts
        if p.area >= small_skinny_area_threshold
    ]

    small_skinny_parts = [
        p for p in skinny_parts
        if p.area < small_skinny_area_threshold
    ]

    discarded_area = sum(p.area for p in small_skinny_parts)
    if discarded_area > 0:
        print(
            "Warning: direct_discard excluded "
            f"{len(small_skinny_parts)} small skinny polygons from both "
            "final classes; "
            f"discarded_area={discarded_area:.6f} m^2 "
            f"({discarded_area / original_area:.6%} of parent polygon)"
        )

    skinny = (
        clean_geom(unary_union(large_skinny_parts))
        if large_skinny_parts
        else None
    )

    return fleshy, skinny


def cleanup_iterative(original_poly, candidate_skinny_raw_parts, config):
    """Iteratively move small polygons to the opposite class.

    Small skinny pieces are merged into fleshy.
    Small fleshy pieces are merged into skinny.

    This preserves original coverage during iteration.
    Remaining small pieces can optionally be discarded after the iteration.
    """
    original_poly = clean_geom(original_poly)
    if original_poly is None:
        return None, None

    original_area = original_poly.area

    small_skinny_area_threshold = (
        config.small_skinny_area_ratio * original_area
    )
    small_fleshy_area_threshold = (
        config.small_fleshy_area_ratio * original_area
    )

    if len(candidate_skinny_raw_parts) == 0:
        skinny = None
        fleshy = original_poly
    else:
        skinny = clean_geom(unary_union(candidate_skinny_raw_parts))
        fleshy = clean_geom(original_poly.difference(skinny))

    for _cleanup_iter in range(config.max_cleanup_iter):
        changed = False

        skinny_parts = explode_to_polygons(skinny)
        large_skinny_parts = [
            p for p in skinny_parts
            if p.area >= small_skinny_area_threshold
        ]

        if len(large_skinny_parts) != len(skinny_parts):
            changed = True

        if len(large_skinny_parts) == 0:
            skinny = None
            fleshy = original_poly
        else:
            skinny = clean_geom(unary_union(large_skinny_parts))
            fleshy = clean_geom(original_poly.difference(skinny))

        fleshy_parts = explode_to_polygons(fleshy)
        large_fleshy_parts = [
            p for p in fleshy_parts
            if p.area >= small_fleshy_area_threshold
        ]

        if len(large_fleshy_parts) != len(fleshy_parts):
            changed = True

        if len(large_fleshy_parts) == 0:
            fleshy = None
            skinny = original_poly
        else:
            fleshy = clean_geom(unary_union(large_fleshy_parts))
            skinny = clean_geom(original_poly.difference(fleshy))

        if not changed:
            break

    final_fleshy_parts = explode_to_polygons(fleshy)
    final_skinny_parts = explode_to_polygons(skinny)

    remaining_small_fleshy = [
        p for p in final_fleshy_parts
        if p.area < small_fleshy_area_threshold
    ]
    remaining_small_skinny = [
        p for p in final_skinny_parts
        if p.area < small_skinny_area_threshold
    ]

    if len(remaining_small_fleshy) > 0 or len(remaining_small_skinny) > 0:
        print(
            "Warning: small polygons remain after iterative cleanup: "
            f"{len(remaining_small_fleshy)} fleshy, "
            f"{len(remaining_small_skinny)} skinny"
        )

        if config.discard_remaining_small_parts:
            large_fleshy_parts = [
                p for p in final_fleshy_parts
                if p.area >= small_fleshy_area_threshold
            ]
            large_skinny_parts = [
                p for p in final_skinny_parts
                if p.area >= small_skinny_area_threshold
            ]

            discarded_area = (
                sum(p.area for p in remaining_small_fleshy)
                + sum(p.area for p in remaining_small_skinny)
            )

            if discarded_area > 0:
                print(
                    "Warning: discarded remaining small polygons after "
                    f"{config.max_cleanup_iter} cleanup iterations; "
                    f"discarded_area={discarded_area:.6f} m^2 "
                    f"({discarded_area / original_area:.6%} of parent polygon)"
                )

            fleshy = (
                clean_geom(unary_union(large_fleshy_parts))
                if large_fleshy_parts
                else None
            )
            skinny = (
                clean_geom(unary_union(large_skinny_parts))
                if large_skinny_parts
                else None
            )

    return fleshy, skinny


def fleshy_paving_resolution(min_dim, config):
    """Return paving resolution for a fleshy polygon."""
    if not np.isfinite(min_dim):
        return config.default_fleshy_paving_resolution

    if min_dim < config.fleshy_resolution_threshold:
        return config.small_fleshy_resolution_factor * min_dim

    return config.default_fleshy_paving_resolution


def decompose_marsh_polygon(original_poly, config):
    """Decompose one marsh polygon into fleshy and skinny parts."""

    original_poly = clean_geom(original_poly)
    if original_poly is None:
        return DecompositionResult([], [], [], [], [], [], [])

    filtered_poly = build_filtered_polygon_for_distance(original_poly, config)
    if filtered_poly is None:
        return DecompositionResult([], [], [], [], [], [], [])

    filtered_parts = explode_to_polygons(filtered_poly)
    filtered_parts = filter_small_polygons(
        filtered_parts,
        config.filter_min_area,
    )

    # 1. Compute fleshy core on the filtered geometry.
    core = filtered_poly.buffer(
        -config.effective_fleshy_core_dist,
        join_style=config.buffer_join_style,
    )
    core = clean_geom(core)

    if core is None or core.is_empty:
        fleshy_mask = None
        candidate_skinny_mask = filtered_poly
    else:
        # 2. Expand core back outward and clip to filtered_poly.
        fleshy_mask = core.buffer(
            config.effective_fleshy_core_dist,
            join_style=config.buffer_join_style,
        )

        fleshy_mask = clean_geom(fleshy_mask)

        if fleshy_mask is None or fleshy_mask.is_empty:
            candidate_skinny_mask = filtered_poly
        else:
            fleshy_mask = fleshy_mask.intersection(filtered_poly)
            fleshy_mask = clean_geom(fleshy_mask)

            candidate_skinny_mask = filtered_poly.difference(fleshy_mask)
            candidate_skinny_mask = clean_geom(candidate_skinny_mask)

    # 3. Map candidate skinny mask back to original geometry.
    if candidate_skinny_mask is None or candidate_skinny_mask.is_empty:
        candidate_skinny_raw = None
    else:
        candidate_skinny_raw = original_poly.intersection(
            candidate_skinny_mask
        )
        candidate_skinny_raw = clean_geom(candidate_skinny_raw)

    candidate_skinny_raw_parts = explode_to_polygons(candidate_skinny_raw)

    # 4. Clean tiny fleshy/skinny pieces.
    if config.small_polygon_cleanup_mode == "iterative":
        final_fleshy, final_skinny = cleanup_iterative(
            original_poly,
            candidate_skinny_raw_parts,
            config,
        )
    elif config.small_polygon_cleanup_mode == "direct_discard":
        final_fleshy, final_skinny = cleanup_direct_discard(
            original_poly,
            candidate_skinny_raw_parts,
            config,
        )
    else:
        raise ValueError(
            "Unknown small_polygon_cleanup_mode: "
            f"{config.small_polygon_cleanup_mode}. "
            "Use 'iterative' or 'direct_discard'."
        )

    fleshy_parts = explode_to_polygons(final_fleshy)
    skinny_parts = explode_to_polygons(final_skinny)
    core_parts = explode_to_polygons(core)
    fleshy_mask_parts = explode_to_polygons(fleshy_mask)
    candidate_skinny_mask_parts = explode_to_polygons(candidate_skinny_mask)

    core_parts = filter_small_polygons(core_parts, config.diagnostic_min_area)
    fleshy_mask_parts = filter_small_polygons(
        fleshy_mask_parts,
        config.diagnostic_min_area,
    )
    candidate_skinny_mask_parts = filter_small_polygons(
        candidate_skinny_mask_parts,
        config.diagnostic_min_area,
    )
    candidate_skinny_raw_parts = filter_small_polygons(
        candidate_skinny_raw_parts,
        config.diagnostic_min_area,
    )

    return DecompositionResult(
        fleshy=fleshy_parts,
        skinny=skinny_parts,
        core=core_parts,
        filtered=filtered_parts,
        fleshy_mask=fleshy_mask_parts,
        candidate_skinny_mask=candidate_skinny_mask_parts,
        candidate_skinny_raw=candidate_skinny_raw_parts,
    )
