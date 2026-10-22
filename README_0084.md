# 0084 — one-command UDF inference, cut, eight-part assembly and real PLY stitching

This is the single entry point requested: `test_udf_sdf_cut_stitch.py`.

**Prerequisites:** Oktopus already has the 0081 UDF curve-local cutter and the 0082 `tools/promote_udf_0081_assembly.py` (your current installation). The 0084 patch includes its own stitching module, so the separate 0083 patch is **not required**.

### Install

From your repository root, after downloading the Git patch:

```bash
git apply --check ~/Downloads/oktopus-0084-one-command-udf-cut-stitch.patch
git apply ~/Downloads/oktopus-0084-one-command-udf-cut-stitch.patch
```

### Single command

```bash
python test_udf_sdf_cut_stitch.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o puffer_0084_udf_cut_stitched_r512 \
  -s oktopus_9_v1 \
  -y udf_adapt_oktopus_puffer_0084_full_pipeline.yaml \
  -r 512 -ckpt final \
  --udf-far-value 0.1 \
  --udf-model-batch-size 32768 \
  --udf-cut-query-batch-size 32768
```

Do **not** supply `--mc-mesh`. The YAML points to the existing 0067 final SDF `riggable_parts/` directory. Nothing regenerates SDF.

### Outputs

In `inference/4M/puffer_0084_udf_cut_stitched_r512/oktopus_9_v1/`:

- `riggable_parts_after_udf_cut/`: eight named final cut `.ply` parts.
- `riggable_after_udf_cut.glb`: eight named cut parts in one GLB scene.
- `riggable_after_udf_cut_UNSTITCHED_parts.ply`: one combined file with no welding.
- `riggable_after_udf_cut_STITCHED.ply`: **written only if** all seven attachment boundaries are bridged and the resulting mesh has one edge-connected component.
- `riggable_after_udf_cut_STITCHED_report.json`: success/failure, topology diagnostics and seam metrics.
- `riggable_after_udf_cut_STITCH_preflight.json`: body/leaf boundary loops, branched edges.
- `root_leaves_ownership.json`, `riggable_after_udf_cut_report.json`, `udf_sdf_cut_stitch_0084_report.json`.

The intermediate full UDF logs, locality evidence and previews remain in
`inference/4M/puffer_0084_udf_cut_stitched_r512/_udf_inference_0084/oktopus_9_v1/`.

### Critical geometry limitation

The stitch stage performs real boundary-loop bridging—not a Boolean union and not mere triangle concatenation. It requires **seven distinct, compatible, simple body attachment loops and one suitable root loop on each tentacle**. The previously observed puffer mesh had branched boundaries; real stitching may therefore fail. In that case this **one command** still saves all eight cut PLYs and the cut GLB, writes a `STITCH_FAILED` JSON explaining why, and exits nonzero. It does **not** create a misleading `STITCHED.ply`.

To tune strict limits centrally, edit only `udf_full_pipeline_0084.stitch` in the YAML. Loosening tolerances does not resolve missing or branched attachment loops; those need a geometric interface-repair algorithm.

### Validation

15 targeted synthetic tests passed, including end-to-end one-command success and failure paths and all eight simple-loop attachment stitches. Real puffer stitching remains unverified until run on the actual eight cut meshes.
