# 0094 — Priority 1: generic certified candidate-ROI pruning

This patch builds conservative **source-localizer** bounding boxes before
model inference for both `rigid_radius` and `wrap_radius` direct adaptations.
It does not change the SDF sign, avatar interval, wrap mapping, blend-group
seam or post-blend Boolean semantics.

## Enable

At the top level of an adaptation YAML, for compatible bounded paths:

```yaml
sdf_candidate_safe_prune_0094: true
sdf_candidate_safe_audit_0094: false
```

`oct22_adapt_oktopus_cone_wrap_0093_sdf_optimization.yaml` now enables the
certified ROI by default. Use the **existing** `test_dc_oct22_editable.sh` runner.
No new runner and no name-based eligibility logic are introduced.

The flag is forwarded to each localizer in the 0093 wrap streaming path,
standard bounded rigid streaming, and eligible staged rigid/wrap paths.
Only bounded direct localization is supported; enabling on an unbounded path
fails early. Adaptations with unknown/modified curve interpolation or invalid
radii fall back to the original complete cylinder broad-phase and print
`[sdf_candidate_safe_0094] fallback=original_broad_phase`.

## Why this is conservative

The stock `PWLACurve` linearly interpolates source center positions and
cylinder radii between its `key_ts`. Its tangent cylinder radius is exactly
1.0. The localizer's acceptance test is an ellipsoid in its reorthogonalized
frame. For each source key interval, the ROI encloses **every** ellipsoid:

- centers and radii use extrema of their two linear key endpoints;
- slowly changing frames use bounds derived from the normalized linear
  interpolation and Gram-Schmidt steps, not sampled approximations;
- rapidly changing frames use a larger orientation-independent spherical bound;
- boxes are padded and intersect the **original** cylinder-box candidate
  union, so they can only remove candidates, never add or reorder them.

These conservative boxes will sometimes save little (especially when the
along-tangent radius is large). No performance win is promised on every
shape. This is an initial geometric broad-phase, **not** the still-pending
candidate-level projection optimization or shared two-section localization.

This is independent of `rigid_radius` versus `wrap_radius`: the radius mapping
happens after the source avatar's localizer gate.

## Validation and profiling

At resolution 128, first run with:

```yaml
sdf_candidate_safe_prune_0094: true
sdf_candidate_safe_audit_0094: true
```

The audit runs the unchanged exact localizer on *excluded* old candidates;
any excluded accepted voxel raises `UNSAFE` **before inference**. It is
expensive and should be off for 512/1024 once geometry is verified. Compare
accepted voxel counts, seam appearance and model output with the flag disabled.
Expected logs include:

```text
[sdf_candidate_safe_0094 audit] full= ... narrow= ... excluded= ... missing_accepted= 0
[sdf_candidate_safe_0094] enabled=True boxes= ... radius_modes=rigid_or_wrap
```

For an unrecognized source implementation, log shows a fallback and the
original candidate sequence is preserved. Do **not** set this simultaneously
with `sdf_candidate_interval_prune`: it is a different, sampled/audited 0048/0057
broad-phase. The two flags intentionally conflict. Disable
`adapt_filter_use_src_interval` when using 0094: the certified ROI must clip
the original full-cylinder boxes, not a truncated cylinder.

### Focused test command

```bash
PYTHONPATH=src:tests python -m unittest \
  tests.test_safe_candidate_roi_0094 \
  tests.test_generic_wrap_0093 \
  tests.test_sdf_candidate_interval_0048 \
  tests.test_candidate_piecewise_0057 -v
```

The synthetic test checks rigid + wrap parity on *identical* accepted voxel
indices and model inputs, candidate reduction, curved-frame enclosures,
fail-open unsupported geometry and an exact-audit failure case. A full GPU
cone result at resolution 1024 still requires a real hardware run.
