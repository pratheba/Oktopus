Oktopus 0044 — root/leaves UDF cut compatible with riggable SDF 0043
========================================================================
Base: Apply on top of 0043 (or 0042 + 0043).

NEW/UPDATED FILES
  src/app/agent_3dvec_udf_cut.py
  src/app/udf_cut_root_leaf_transfer.py (NEW)
  exp/train/manipulation/adapt_oktopus_puffer_root_leaves_udf.yaml (NEW)
  exp/train/manipulation/adapt_oktopus_puffer_blend_udf_cut_root_leaves.yaml (NEW)
  tests/test_udf_cut_root_leaves_0044.py (NEW)

SDF RUN FIRST
  --test_file adapt_oktopus_puffer_root_leaves_udf.yaml
  --surface-extraction mc --sdf-localize-after-gate
  --dcsdd-fast-bbox-marking
  Use a distinct output folder; start at res=512.
  DO NOT pass rigid-only --dcsdd-stream-inference / crop-first / replay.

The new SDF YAML preserves original 0043 puffer geometry and group parameters,
removes an accidental duplicate root_leaves YAML key, and changes ONLY:
  export_individual_meshes: true
This is essential for the existing per-item UDF query/cut evidence.

UDF RUN SECOND
  --test_file adapt_oktopus_puffer_blend_udf_cut_root_leaves.yaml
  Keep your existing --mc-mesh pattern for the individual MC/PLY parts, e.g.:
  inference/4M/<SDF_RUN>/oktopus_9_v1/{index}_{mode}_{accessory}.ply

IMPORTANT: Edit the UDF YAML's
  udf_cut_root_leaves_mc_dir:
to point at the actual SDF run's
  <SDF_RUN>/oktopus_9_v1/riggable_parts
The YAML example folder is illustrative, not auto-discovered.

UDF behavior:
- Scores/cuts each original adaptation unchanged.
- Part transfer evidence from each item is applied only to its owner part,
  as mapped by root_leaves, after intra-tentacle seam blending.
- No global fused MC input is required and `udf_cut_combined_mc_mesh` is removed.
- An object with no UDF removal evidence is kept unchanged.
- Intersecting tentacles remain separate named meshes for rigging.
- Existing UDF thresholds/transfer parameters unchanged.

UDF OUTPUTS (inside UDF output folder):
  riggable_after_udf_cut.glb
  riggable_parts_after_udf_cut/body.ply
  riggable_parts_after_udf_cut/tentacle_t{1,2,3,4,5,6,8}.ply
  riggable_after_udf_cut_report.json

HISTORICAL UDF/SDF YAML DIFFERENCES
The user's provided UDF YAML already has two tloc offsets that are *not*
active in the provided SDF puffer YAML:
  item 8 (t4_2) tloc: [0,0,-0.05]
  item 11 (t5_2) tloc: [0,0,-0.05]
0044 preserves these pre-existing values, but review them if UDF scoring is
spatially displaced relative to the generated SDF meshes.

CONE 0042 NOTE
The cone 512 trace's ValueError is from rigid-only streaming flags, not
wrap_radius/localization. For combined cone wrap, use
  --surface-extraction mc --sdf-localize-after-gate --dcsdd-fast-bbox-marking
and REMOVE --dcsdd-stream-inference / --dcsdd-stream-one-pass /
--dcsdd-stream-crop-first / --dcsdd-stream-replay-metadata /
--dcsdd-stream-accepted-count / --dcsdd-compact-staging.

VALIDATION
Run:
  PYTHONPATH=src:src/app:src/ngc python -m unittest discover -s tests -p 'test_udf_cut_root_leaves_0044.py' -v
  PYTHONPATH=src:src/app:src/ngc python -m unittest discover -s tests -p 'test_root_leaf_combination_0043.py' -v
Synthetic tests verify exact part ownership, fail-fast missing inputs,
unchanged no-evidence part, and GLB roundtrip. Full neural SDF/UDF puffer runs
were not executed in the build environment.
