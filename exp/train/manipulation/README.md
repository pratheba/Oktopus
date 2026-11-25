# Oktopus — two manipulation YAMLs per avatar/accessory pair

**This package has exactly two active YAMLs per new pairing:**

- `sdf/nov25_<pair>_sdf.yaml` — run SDF adaptation; generate the fused exterior, parts, and ownership manifest required by the direct cutter.
- `sdf_udf/nov25_<pair>_sdf_udf.yaml` — provide the SAME adaptation mapping to `tools/udf_run_direct.py`. That tool loads a TRAINED UDF model from `--config` and `--ckpt`, and injects `udf_direct_fused_0100` automatically. **No previous UDF mesh/run is needed.**

There is NO separate `udf/` adaptation folder in the current direct-cutting workflow.

The two TRAINING YAMLs (one joint avatar + accessory SDF, one joint avatar + accessory UDF) remain separately under `config/` and are NOT included here.

## Working precedent

The following files are copied **unchanged** as your known-working Oktopus puffer reference:

- `sdf/oct22_adapt_oktopus_puffer_full_sdf_optimization.yaml`
- `sdf_udf/oct22_adapt_oktopus_puffer_full_udf_fused_0092.yaml`

The 0092 YAML contains historical 0081/0092 options because of how that experiment evolved. **The current `tools/udf_run_direct.py` injects `udf_direct_fused_0100` and no longer requires `--udf-run`.** For the six new pairs, the YAMLs retain aligned adaptation/ownership information; do not blindly enable older 0092/0084 pipelines.

## Install

Extract/copy the folders into `exp/train/manipulation/` (back up existing files first). The inspected `tools/udf_run_direct.py` only accepts flat YAML filenames, so while inside `exp/train/manipulation`, run:

```bash
python make_flat_aliases.py             # preview only
python make_flat_aliases.py --apply     # makes root-level symlinks to files in sdf and sdf_udf
python verify_pairs.py
```

If your SDF inference runner accepts a YAML basename only, the SDF links also handle that.

## Run T-rex dress (example)

First train the TWO joint networks with their own `config/*.yaml` files. For UDF, the dataset root should be `Pack3MDataset/UDF` with **real `udf_samples.pkl` for both avatar and accessory**.

1. SDF inference with `nov25_trex_dress_sdf.yaml` and the matching SDF checkpoint (use your normal SDF inference command and set an output run name, e.g. `trex_dress_sdf_r128`). SDF should write:
   - `inference/4M/trex_dress_sdf_r128/trex_9_v1/riggable_fused_sdf_exterior.ply`
   - `.../riggable_parts/*.ply`
   - `.../root_leaves_ownership.json`

2. After the UDF checkpoint is trained, direct cutting:

```bash
python tools/udf_run_direct.py \
  --sdf-run trex_dress_sdf_r128 \
  --out trex_dress_udf_r128 \
  --shape trex_9_v1 --model-size 4M --resolution 128 \
  --config config/config_udf_grid_b13d3_trex1_dress2_v2.yaml \
  --yaml nov25_trex_dress_sdf_udf.yaml \
  --ckpt final --parts all \
  --allow-uncalibrated-nonbody
```

Set `--config` to the **actual trained joint UDF config**, and use the checkpoint you actually saved. Replace `--sdf-run` with the actual completed SDF run name.

**Status:** Six new mappings are **candidate configs**, not validated geometrically. They were copied from earlier generated pairs and checked for exact SDF/UDF adaptation and ownership equality. The Oktopus puffer reference is unchanged. Check `--dry-run`, manifest, and actual mesh outputs before costly 1024 or 2048 inference. `--allow-uncalibrated-nonbody` is experimental.

**Note:** `sdf_fused_exterior_0087: true` in the SDF file requests the fused output. Depending on your current SDF inference mode, this may require the DCSDD `ours` extractor and a disk-backed grid. The YAML alone does not guarantee those meshes are produced.
