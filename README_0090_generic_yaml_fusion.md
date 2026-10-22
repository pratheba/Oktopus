# 0090: YAML-controlled per-part and global SDF fusion

This change is an **SDF** change. It is not a generic UDF cutter and does **not**
make a stitched or watertight final surface automatically.

There are three independent decisions:

1. **Radius mode** per adaptation: `wrap_radius` XOR `rigid_radius`.
2. **Per-part SDF composition**: give several adaptation records the **same**
   `blend_group` and assign that group under `root_leaves.parts`. The original
   group's seam settings are preserved. One record can define a standalone part.
3. **Global Boolean**: `sdf_fused_exterior_0087: true` makes a global scalar
   min-union of the finished parts, in addition to the separate riggable parts.
   `false` keeps the parts separate. `combine_adaptations: true` is required.

For example (body and one sleeve with two blended adaptations):

```yaml
combine_adaptations: true
export_individual_meshes: false
sdf_fused_exterior_0087: true  # optional global min-union
root_leaves:
  output_mode: separate_parts
  export_legacy_combined: false
  parts:
    body:
      target_key: trex_9_v2|body
    sleeve:
      blend_group: sleeve_group
      attach_to: body  # ownership metadata only, NOT welded topology
adaptations:
  - target_key: trex_9_v2|body
    wrap_radius: true
    rigid_radius: false
    # ...original adaptation mapping...
  - target_key: trex_9_v2|front_l_leg
    blend_group: sleeve_group
    wrap_radius: false
    rigid_radius: true
    # ...original adaptation mapping...
  - target_key: trex_9_v2|front_l_leg
    blend_group: sleeve_group
    wrap_radius: false
    rigid_radius: true
    # ...original adaptation mapping...
```

`root_leaves.parts` also accepts `adaptation_name` or `adaptation_indices: [0, 1]`.
Every adaptation must be assigned exactly once. **Do not** put alternative,
mutually exclusive experiments in the same group unless you actually want to
union their SDFs. Do not reuse a blend group across two output parts.

The legacy `root_leaves: {root: ..., leaves: ...}` format (the 0067 puffer)
continues to work unchanged.

## Run

`test_dc_oct22.sh` still invokes `tools/oct22_run.py sdf`; pass a YAML under
`exp/train/manipulation`, and an actual model config for the target shape.
`--require-fused` checks `sdf_fused_exterior_0087: true` before inference.

Oktopus boot (one rigid adaptation; fusion is a no-op on one solid but
provides a conventional fused PLY file):

```bash
python tools/oct22_run.py sdf \
  --case oktopus_boots_t7_boot1_only \
  --yaml oct22_example_oktopus_boots_t7_fused_sdf.yaml \
  --shape oktopus_9_v1 \
  --resolution 128 --require-fused \
  --out boots_t7_fused_r128 --dry-run
```

Cone (two wrap adaptations in one `blend_group`):

```bash
python tools/oct22_run.py sdf \
  --case oktopus_cone_post_sdf \
  --yaml oct22_example_oktopus_cone_fused_sdf.yaml \
  --shape oktopus_9_v1 \
  --resolution 128 --require-fused \
  --out cone_fused_r128 --dry-run
```

T-Rex (mixed or multiple SDF adaptations on named parts):

```bash
python tools/oct22_run.py sdf \
  --case trex2_puffer \
  --yaml oct22_example_trex2_puffer_fused_sdf.yaml \
  --shape trex_9_v2 \
  --config config/YOUR_TREX_MODEL_CONFIG.yaml \
  --resolution 128 --require-fused \
  --out trex2_puffer_fused_r128 --dry-run
```

T-Rex dress (sample selection: first top and bottom; **other entries omitted**):

```bash
python tools/oct22_run.py sdf \
  --case trex_dress \
  --yaml oct22_example_trex_dress_selected_fused_sdf.yaml \
  --shape trex_9_v1 \
  --config config/YOUR_TREX_MODEL_CONFIG.yaml \
  --resolution 128 --require-fused \
  --out trex_dress_selected_r128 --dry-run
```

Armadillo boots (explicitly selected shaft + foot, one fused boot):

```bash
python tools/oct22_run.py sdf \
  --case armadillo_boots2 \
  --yaml oct22_example_armadillo_boots2_selected_fused_sdf.yaml \
  --shape armadillo_13_v1 \
  --config config/YOUR_ARMADILLO_MODEL_CONFIG.yaml \
  --resolution 128 --require-fused \
  --out armadillo_boots2_selected_r128 --dry-run
```

This example intentionally selects **only** source adaptation indices 0 and 3;
indices 1 and 2 were alternative shaft configurations. Check that selection
matches what you want before running.

Armadillo dress: use `--case
armadillo_dress`, `--shape armadillo_13_v1`, and `--config` pointing at
**your actual corresponding model config**. Original snapshots contain
experimental alternatives: decide intended groups and which alternatives to
use before enabling fusion. Don't assign variants to parts blindly.

Remove `--dry-run` when ready. For 0087 fusion keep `--sdf-grid-memmap-dir`
(as set by the runner), and use a **new** `--out` each time.

Outputs with named parts:

- `riggable_parts/<part>.ply`: SDF-extracted individual part. If its selector is
  a `blend_group`, the internal part field was merged before extraction.
- `*_riggable.glb`: unstitched multi-part GLB, with part names from YAML.
- `root_leaves_ownership.json`: adaptation ownership and `attachments` metadata.
- `riggable_fused_sdf_exterior.ply`: global scalar-min-union *only if enabled*.
- `riggable_fused_sdf_exterior_report.json`: diagnostics, not a watertightness guarantee.

With no `root_leaves` and `combine_adaptations: true`, 0087 may export the
combined scalar union directly, but it cannot preserve separate part identity.

**UDF warning:** `test_udf_sdf_cut_stitch.py` / 0084 and `test_udf_oct22.sh` are
still puffer-specific eight-part workflows. They are not generic fused-shell
UDF runners; this patch makes **SDF** routing generic, not UDF cutting.
