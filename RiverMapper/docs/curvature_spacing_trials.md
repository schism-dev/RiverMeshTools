# Curvature source for river-arc spacing

## Status: on hold

The FID 11 investigation and temporary comparisons were completed on
2026-10-12. Curvature changes have not been integrated into RiverMapper.
Further implementation is on hold.

**Thalweg turns can be important even when the river banks are smooth.** Dense
vertices are justified when the arcs need to follow those turns. Bank-midpoint
curvature is an alternative for representing the overall channel shape; fewer
vertices alone do not establish a better result. Preserve both curvature sources
as options if this work resumes. No change of the default has been decided.

## Motivation and existing behavior

The reported location is longitude -123.94974124, latitude 46.31032937, near
FID 11 in `USGS_pyDEM_clipped_w_area_line_explode_ll.shp`. FID here is the
zero-based feature index.

`redistribute_arc` in `RiverMapper/make_river_map.py` determines spacing from
thalweg curvature and channel width:

```python
spacing = clip(
    min(R_coef / curvature,
        length_width_ratio * channel_width / (nrow_arcs - 1)),
    minimum_spacing,
    maximum_spacing,
)
```

The current curvature calculation uses `moving_average(..., n=100,
self_weights=2)` and takes the maximum curvature from full and alternate-vertex
samples. Two effects were identified:

- The smoother retains two-thirds of the original coordinates, allowing small
  thalweg bumps to drive spacing to the minimum. At the reported location,
  estimated bend radii around 10–40 CPP m can request 5 CPP m spacing, while the
  width constraint permits roughly 45–60 CPP m.
- At either end of a line longer than 100 vertices, the moving average is held
  constant for 50 vertices. The weighted result there is
  `(constant + 2 * original) / 3`, which shrinks the geometry rather than
  smoothing its turns. Curvature increases by a factor of 1.5. This was confirmed
  at the reported location during the second redistribution pass.

These numerical effects should be distinguished from the choice to follow real
thalweg turns: smooth banks do not make those turns irrelevant.

## Temporary options tested

Both experimental sources use a local quadratic fit with weights proportional
to along-thalweg distance represented by each sample. Derivatives of the fit
provide curvature. The fitted coordinates are used for curvature estimation;
redistribution retains a subset of the actual thalweg coordinates. Subsequent
thalweg correction, smoothing, and bank searches still run normally.

The starting fit window is **3W** in total, centered at the vertex where
possible, with a one-sided window near endpoints. `W` is the local median of
resolved, unclamped bank widths; the trial uses 5 CPP m where that local estimate
is unavailable. The existing spacing/deletion rule and its per-vertex channel
width constraint are retained. Endpoints and pinned vertices are retained.

| Experimental source | Fitted coordinates | Sparse or unreliable samples |
|---|---|---|
| `thalweg` | Thalweg coordinates | Expand to the three nearest distinct positions if the window contains fewer than three. |
| `bank_midpoint` | `(left_bank + right_bank) / 2` from the bank search immediately preceding redistribution | Fit only points with both banks resolved and width above the minimum clamp. Fall back to the thalweg fit if fewer than three reliable distinct positions occur in the window. |

Bank-midpoint fits use the same along-thalweg distance to define their windows.
They do not use the final cleaned or nudged bank geometry, which is produced
later in the pipeline. No combined curvature rule was tested.

## FID 11 comparison

Input directory: `/sciclone/schism10/feiye/Charles_RiverMapper/chinook`.
Inputs: `USGS_chinook_wgs84_edited.tif` and
`USGS_pyDEM_clipped_w_area_line_explode_ll.shp`.

Common settings were `selected_thalweg=[11]`, `nhd_area_tif=False`,
`river_threshold=(5, 50)`, `width2narcs_option='sensitive'`, `R_coef=0.1`,
`along_channel_reso_thres=(5, 100)`, `pseudo_channel_width=5`, and
`pseudo_channel_dl=30`.

All comparison runs temporarily accepted bank-search status `-3` (search-limit
failure), using the earlier real-bank comparison setup. This prevented
whole-line pseudo fallback from hiding the curvature effects. Other quality
rules remained active. Selective pseudo stretches were not applied.

| Method | First redistribution vertices | Final thalweg vertices | Final vertices within 100 UTM m of the reported location | Output arc features |
|---|---:|---:|---:|---:|
| Current algorithm | 864 | 538 | 13 | 82 |
| Width-scaled thalweg fit | 702 | 513 | 9 | 56 |
| Width-scaled bank-midpoint fit | 735 | 525 | 7 | 69 |

All runs retained 30 pinned vertices and generated real-bank products without
whole-line pseudo fallback. The bank-midpoint fit was used at 2,379 of 3,237
vertices in the first pass and 267 of 735 in the second; the remaining vertices
used the thalweg fit. Thus this was a bank-midpoint option with a fallback,
not a comparison using bank curvature everywhere.

The bank-midpoint option reduced density at the reported location but retained
more vertices over the full reach than the thalweg option. Larger bends remained
recognizable in the comparison plot. This single-feature result does not establish
a preferred default or demonstrate sufficient resolution for every bed feature.

Checks performed: analytic straight and circular lines, straight bank midpoints
with a wiggly thalweg, retained-coordinate equality during redistribution,
endpoint and pinned-vertex retention, output geometry validity, source-file
hashes, and inspection of comparison figures. Existing bank-search failures and
the thalweg-smoothing convergence warning remained.

Algorithm distances and widths above are in RiverMapper's CPP coordinates.
The 100 m neighborhood counts and plot axes use UTM zone 10N distances; these
should not be interpreted as identical distance measures.

## Saved artifacts and resumption

Temporary runners in this checkout:

- `Untracked_temp_files/bend_diagnostics/trial_width_scaled.py`
- `Untracked_temp_files/bend_diagnostics/trial_bank_midpoint.py`

Results and summaries are in the sibling `width_scaled_trial/` and
`bank_midpoint_trial/` directories. These are local, untracked artifacts.
The ZIP products also include the runner, summary, comparison plot, map files,
and shapefiles, so the trial record is preserved with the exported product.

Copies in `~/s1/Charles_RiverMapper/chinook/` (the input directory above):

- `fid11_width_scaled_trial.map`, `.zip`, `_comparison.png`, and
  `fid11_width_scaled_trial_shapefiles/`
- `fid11_bank_midpoint_trial.map`, `.zip`, `_comparison.png`, and
  `fid11_bank_midpoint_trial_shapefiles/`

If resumed, retain an explicit curvature-source choice such as `thalweg` and
`bank_midpoint`. Evaluate both against the intended need to follow thalweg turns
as well as channel shape, particularly at intersections, tight bends, and
stretches with unreliable banks. The fit-window ratio and fallback behavior
remain experimental. The temporary search-status override is part of the
comparison setup, not a proposed production change.
