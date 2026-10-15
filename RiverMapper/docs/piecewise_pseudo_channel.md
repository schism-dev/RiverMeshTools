# Piecewise pseudo-channel fallback

## Integration status

Merged into `main` on 2026-10-12 as `be1d483` from
`feature/shallow-river-depth-param`. The feature branch remains available for
comparison. The two latest feature commits are:

- `9213b5d`: partial lateral DEM coverage, thalweg-correction fixes, focused tests,
  and documentation of the paused curvature trials.
- `1b8c7c2`: optional piecewise pseudo-channel fallback, focused tests, and usage
  documentation.

The merge also includes the earlier shallow-depth retries and handling of kept
thalwegs after bank-search failures. Merging does not enable the piecewise
option automatically.

To compare the piecewise change with its preceding commit, run from the
`RiverMapper` project directory:

```bash
git diff --stat 9213b5d 1b8c7c2
git diff 9213b5d 1b8c7c2 -- \
  README.md \
  RiverMapper/config_river_map.py \
  RiverMapper/make_river_map.py \
  RiverMapper/Tests/test_piecewise_pseudo_channel.py \
  docs/piecewise_pseudo_channel.md
```

## Enable the option

```python
from RiverMapper.config_river_map import ConfigRiverMap

config = ConfigRiverMap(
    i_pseudo_channel=2,
    i_piecewise_pseudo_channel=True,
    pseudo_channel_width=5,
    piecewise_pseudo_channel_ratios=(5, 3, 1),
)
make_river_map(..., **config.optional)
```

The same keyword arguments can be passed directly to `make_river_map`.
The option defaults to **disabled** and requires `i_pseudo_channel=2`.
Curvature-based thalweg redistribution is unchanged; the curvature-source trials
remain [on hold](curvature_spacing_trials.md).

For the direct call in `chinook/process_CRE.py`, add these arguments to the
existing `make_river_map` call:

```python
i_piecewise_pseudo_channel=True,
piecewise_pseudo_channel_ratios=(5, 3, 1),
```

Keep the existing `pseudo_channel_width=5` and `pseudo_channel_dl=30` settings.
`pseudo_channel_dl` continues to govern whole-line pseudo redistribution; it
does not set the width-scaled piecewise selection window or transition length.
The script inspected on 2026-10-12 omitted `i_piecewise_pseudo_channel`, so its
full run used the disabled setting and retained FID 11 as a whole-line pseudo
channel. The enabled FID 11 test used the flag explicitly.

## Selection and blending

After bank nudging and optional bank smoothing, compute absolute turning angles
at the thalweg and both banks. Excess turning is:

```python
excess = maximum(maximum(left_turn, right_turn) - thalweg_turn, 0)
```

`W` estimates nearby channel width from bank points whose excess turning is at
most 0.5 radians and whose width is greater than `pseudo_channel_width`.
These low-turn points are a geometric proxy for usable banks, not a guarantee
that bank search resolved both sides. Their widths are interpolated through
rough stretches, then replaced by a nearby median where samples are available.
If there are no usable width samples, `W` is `pseudo_channel_width`.

The three ratios control:

| Ratio | Default | Meaning |
|---|---:|---|
| Roughness window | 5 | Average excess turning over a distance window of 5W. |
| Minimum stretch | 3 | Keep runs above 0.5 radians only when their integrated length `integral(ds/W)` is at least 3. |
| Transition | 1 | Blend over one unit of `integral(ds/W)` outside either end of each selected run. |

The roughness average is weighted by along-thalweg distance, rather than vertex
count. The transitions use smoothstep weights `r*r*(3-2*r)`; overlapping weights
are combined by taking their maximum. All ratios must be positive and finite.
Distances and widths use RiverMapper's CPP coordinates.

When the option is enabled, a tuple with the wrong length, nonpositive values,
NaN, or infinity raises `ValueError`. This is configuration validation before
processing, not a per-linestring rejection or a response to difficult terrain.
The default `(5, 3, 1)` is valid.

Selected cores use fake banks of exactly `pseudo_channel_width` around the
existing thalweg. Transition points blend the real and fake banks. Banks at
zero-weight vertices are unchanged. No new thalweg vertices are added or moved
by this helper. Core width remains an explicit configured width, rather than a
multiple of the reference width.

## Interaction with bank-search and quality rules

With the option enabled, final search-limit status `-3` can proceed through the
normal bank touch-ups and piecewise blending instead of immediately replacing
the entire line with a pseudo channel. This permits usable stretches to retain
their real-bank geometry even when a few transects never find land.

The earlier `keep` checks and DEM coverage requirements remain active: missing
thalweg elevations are rejected, lateral rays stop at coverage gaps, and
`keep=0` lines with search-limit failures are still rejected by the earlier
checks. Narrow-channel status `-1` retains the existing whole-line fallback.
The final river-arc quality check also retains its whole-line pseudo fallback.

Because the selector measures geometry, a search-limit failure by itself does
not force a local pseudo stretch. Smooth bank-search endpoints can remain as
real-bank estimates. Use the opt-in setting where that behavior is appropriate.

### Full chinook run: missing thalweg coverage

The 2026-10-12 12:09 run of `process_CRE.py` rejected 41 of 735 input features,
leaving 694 in `Outputs/valid_thalwegs.shp`. These were exactly the same FIDs
rejected by the preceding full run on 2026-10-11 at 20:01. All rejections were
caused by missing DEM elevations on the original thalweg; they were not new
piecewise-fallback exclusions. All input features had `keep=1`.

The rejected coordinates are within the TIFF's rectangular bounds but encounter
NoData in its valid-data footprint. Seventeen features have no valid sampled
thalweg elevations; 24 have partial coverage. FIDs 487, 518, and 627 each have
NoData at only their first two vertices:

| FID (zero-based) | Total vertices | Missing vertices |
|---|---:|---:|
| 487 | 941 | 2 |
| 518 | 67 | 2 |
| 627 | 24 | 2 |

Those samples were checked directly against the TIFF and matched its NoData
sentinel and cached DEM. An endpoint repair for these cases has not been
implemented. The complete-thalweg-coverage requirement still applies, even with
`keep=1`. FID 11 differs because its original thalweg has complete coverage;
its earlier coverage problem occurred along lateral bank-search rays.

## FID 11 test

The production API was tested with the chinook inputs and the existing settings
`river_threshold=(5, 50)`, `R_coef=0.1`,
`along_channel_reso_thres=(5, 100)`, `width2narcs_option='sensitive'`,
`pseudo_channel_width=5`, and `pseudo_channel_dl=30`.
Bank-search and curvature functions were not replaced in this test.

| Setting | Whole-line pseudo fallback | Output arc features |
|---|---:|---:|
| Disabled | Yes | 18 |
| Enabled, ratios (5, 3, 1) | No | 31 |

The enabled run selected 12 pseudo cores covering about 19.0% of the CPP
thalweg length. It kept 538 thalweg vertices, with 122 core vertices and
131 vertices receiving any blending. Both ends were preserved, zero-weight
banks matched their pre-blend coordinates exactly, core widths were 5 CPP m,
and exported geometries passed validity checks. Existing bank-search failures
and the thalweg-smoothing convergence warning remained.

The width-scaled selector is more fragmented than the earlier temporary trial
using 21 vertices, a 100 CPP m minimum run, and a 30 CPP m transition, which
selected three cores. The ratios are starting settings and can be adjusted
without changing code; this single-line test does not establish universal
defaults.

Reproducible runner and local outputs:

- `Untracked_temp_files/chinook_feature11_diagnostics/test_piecewise_production.py`
- `Untracked_temp_files/chinook_feature11_diagnostics/piecewise_production_trial/`

Review copies in `~/s1/Charles_RiverMapper/chinook/`:

- `fid11_piecewise_pseudo_channel.map` and `.zip`
- `fid11_piecewise_pseudo_channel_comparison.png`
- `fid11_piecewise_pseudo_channel_shapefiles/`

The ZIP includes the runner, summary, comparison figure, and enabled-run products.
Focused unit checks are available with:

```bash
python -m unittest RiverMapper.Tests.test_bank_dem_coverage RiverMapper.Tests.test_piecewise_pseudo_channel -v
```
