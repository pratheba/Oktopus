# Oktopus OCT22 — paired SDF/UDF adaptation settings

Contents: `exp/train/manipulation/oct22_adapt_<experiment>_sdf_optimization.yaml` and `oct22_adapt_<experiment>_udf_optimization.yaml` plus original uploaded source snapshots and an explicit manifest.

## Ready for the existing FINAL pipeline: Oktopus full puffer

- `oct22_adapt_oktopus_puffer_full_sdf_optimization.yaml`: **exact 0067 geometry/adaptation settings** (15 source adaptations, 8 root/leaf parts), audited performance flags, `cut_avatar: false`, no individual intermediate meshes. This is the retained SDF baseline.
- `oct22_adapt_oktopus_puffer_full_udf_optimization.yaml`: **exact uploaded 0084 UDF settings** aligned with the 0067 SDF adaptation records and root/leaf ownership. Supports independent calibrated UDF, 0081 curve-local 4-region expansion, separated eight-part output, assembly and conservative stitching attempt via `test_udf_sdf_cut_stitch.py`.
- UDF points to `inference/4M/puffer_0067_final_parts_r512/oktopus_9_v1/riggable_parts`. If the SDF output was written under a different run folder, update `udf_cut_root_leaves_mesh_dir`; don't regenerate SDF unnecessarily.
- The 0084 entry point requires code patches through 0084 **plus the 0085 missing-module fix**; successful stitching still depends on the actual attachment boundaries.

Example run, using the original SDF output directory:

```bash
python test_udf_sdf_cut_stitch.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o puffer_oct22_udf_cut_stitched_r512 \
  -s oktopus_9_v1 \
  -y oct22_adapt_oktopus_puffer_full_udf_optimization.yaml \
  -r 512 -ckpt final --udf-far-value 0.1 \
  --udf-model-batch-size 32768 --udf-cut-query-batch-size 32768
```

## Imported boots, cone, armadillo and T-Rex experiments

Each uploaded YAML gets a matching SDF and UDF **snapshot**. The source's target keys, accessory keys, transforms, curve intervals, blending and per-adaptation parameters remain paired and intact. Original uploads, including commented experiments, are retained under `source_snapshots/`.

The SDF configs add common 0067-era performance flags but deliberately **disable unvalidated body-specialized accelerators** (0057/0058/0059/0062). We also set `cut_avatar: false` in the copied adaptations so UDF is intended to own the removal. These are *proposed conversions*, not measured-equivalent final settings for those geometries. `oct22_manifest.json` documents every changed flag.

**Do not use the generic UDF snapshots with `test_udf_sdf_cut_stitch.py` yet.** That runner currently requires a puffer-specific 15-to-8 ownership manifest, root, and seven leaf attachments. The other uploaded YAMLs have no guaranteed compatible final SDF part manifest or matching root/leaf stitching plan. Their UDF files contain matching adaptation records and safe common calibration defaults as a starting point only. A generic single-/multi-part UDF ownership and final-export path must be implemented and validated before they are runnable in that pipeline. No fictitious input mesh directories or attachment graphs have been invented.

Files containing several adaptations for the **same target key** may be collections of historical experiments, not intended concurrent adaptations. Their blocks are preserved for inspection; do not assume all should run together.

## Verification

- All generated YAML files parse using PyYAML.
- Every paired SDF/UDF adaptation array is structurally identical for its experiment.
- Puffer SDF/UDF source inputs both have 15 equivalent adaptation entries, identical `blend_groups` and `root_leaves`.
- No source file has been overwritten.
- No SDF/UDF inference or geometry regression was run for the converted experiments.
