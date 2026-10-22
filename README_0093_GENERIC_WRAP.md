# 0093 · Safe wrap optimizations and post-blend avatar Boolean

**Generic**: no cone/puffer/T-Rex names are used for eligibility. Applies to
all supported `wrap_radius` adaptations, including wrap sections in mixed
rigid+wrap YAMLs. Geometry-specific checks, not accessory names, select a path.

## What happens automatically

`src/app/agent_3dvec_sdf.py` now defaults `sdf_wrap_auto_safe_0093: true`.
For each adaptation independently, it enables the existing 0045 zero-angular
shortcut only when `can_skip_inert_snug` proves the automatic snug field has
no effect (`direct`, wrap-only, `wrap_angular_follow: 0`, multiplicative snug,
no provided shared field, no active attachment blend/guard/debug). Rigid
adaptations or ineligible wrap adaptations remain unchanged. Set
`sdf_wrap_auto_safe_0093: false` globally to disable, or set
`sdf_optimize_zero_angular_wrap: false` on an adaptation to opt out locally.
This does *not* enable 0057/0058/0059/0062 pruning indiscriminately.

## Optional high-resolution streaming group

`sdf_stream_combined_wrap: true` is available to **2 or more compatible direct
wrap sections** sharing the *same target curve and blend_group*, with a
contiguous post-SDF overlap crossfade seam chain, zero angular follow, no
active snug/offset/attach-to-free geometry changes. The named output part is
defined by `root_leaves.parts`. It can save that part and an identical
single-part fused exterior (`sdf_fused_exterior_0087: true`) without a
full-resolution second accumulator.

This is NOT yet a general streaming engine for multi-target, mixed
rigid+wrap or multi-part Boolean fusion; those retain the existing standard
path. Unsupported combinations fail with explicit errors rather than
silently changing geometry.

### Post-blend avatar subtraction (optional)

```yaml
sdf_postblend_avatar_boolean_0093:
  enabled: true
  clearance: 0.0001
  avatar_overlap_tolerance: 0.005
```

This applies `max(SDF_blended, clearance - SDF_avatar)` **after** all
within-part seam crossfades, using avatar SDF predicted at the **same global
voxel indices** during the one-pass per-section traversal. Overlapping avatar
samples must agree to within the configured tolerance or processing fails.
The output is a new *geometry variant* compared with `cut_avatar: true` on
individual sections. Keep `cut_avatar: false` per section when using it.

For a mixed group or more than one root/leaf part, do not set the post-blend
flag; the code rejects unsupported cases. They continue to use their existing
per-adaptation cut semantics until a generalized postblend path is implemented.

### Sections and outputs

Setting `sdf_stream_export_sections_0093: true` saves each uncarved section to
`individual_adaptations/section_00.ply`, `section_01.ply`, etc. (diagnostic).
The final cut/blended `riggable_parts/<part>.ply` is a *separate* output.
Setting `sdf_fused_exterior_0087: true` also saves
`riggable_fused_sdf_exterior.ply` and its report. With **one final part**, this
fused file is the same surface; no unnecessary second global min-union grid.

## Run cone first at 128

Patch 0093 is applied **on top of 0092**. The new cone example YAML is
`exp/train/manipulation/oct22_adapt_oktopus_cone_wrap_0093_sdf_optimization.yaml`.
Use the existing editable script; there is no new script to maintain:

```bash
# In test_dc_oct22_editable.sh set:
CASE="cone"
RESOLUTION=128
OUT="cone_wrap_0093_r128"
EXPECT_FUSED=1
DRY_RUN=0

bash test_dc_oct22_editable.sh
```

The script's `cone` case now selects the new 0093 YAML. Confirm logs include
`[sdf_stream_combined_wrap] enabled=True`, then
`[sdf stream 0093] POSTBLEND_AVATAR_BOOLEAN applied`, and
`[sdf fused 0087] saved .../riggable_fused_sdf_exterior.ply`.
Inspect the blended seam and Boolean cavity at 128 before 512/1024.

## T-Rex, dresses, mixed cases

Continue using the existing October 22 YAML with
`bash test_dc_oct22_editable.sh` and the appropriate `CASE`. The **safe per-item
zero-angular optimization** is automatically considered. Do **not** add
`sdf_stream_combined_wrap: true` or the post-blend flag to mixed/multi-target
configurations: those are not yet supported for this shortcut.

## Verification / limits

The old two-section streaming regression, multi-section synthetic SDF seam,
root/leaf export, avatar-alignment and Boolean arithmetic are covered by tests.
Full 1024 GPU geometry/resource parity on cone and T-Rex is not proven. This
patch deliberately fails on unsupported feature combinations rather than
silently changing normals, snugging, cutting or blend semantics.
