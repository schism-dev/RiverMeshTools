# Piecewise pseudo-channel fallback

## Enable the option

```python
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
