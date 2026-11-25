# Oktopus manipulation configurations — organized 2026-11-25

The original uploaded `manipulation.zip` is preserved without changes in `archive/`. This package does not delete or overwrite your checkout.

| Folder | Role | Runner |
|---|---|---|
| `sdf/` | Avatar/accessory SDF adaptation | SDF inference / `-y` |
| `udf/` | Legacy standalone UDF optimization/adaptation experiments | Old UDF-specific runner, **not** 0107 |
| `sdf_udf/` | UDF model queries on completed SDF output | `tools/udf_run_direct.py --yaml ...` |
| `sdf/paired_direct/` | SDF YAML with named part ownership + fused exterior required for NEW 0107 pairings | SDF inference |
| `archive/` | Complete original 196-YAML archive, untouched | None |

## Essential distinction

- `config/config_*sdf*.yaml` and `config/config_*udf*.yaml` TRAIN the networks jointly on avatar + accessory. **They are not manipulation YAMLs.**
- An `sdf_udf/` YAML does not train a UDF and is not a separately meshed UDF. It provides exactly the SDF adaptations + ownership layout while the 0107 runner loads the UDF checkpoint via `--config` / `--ckpt`.
- Legacy `udf/` optimization configs (`udf_cut_seed_world`, etc.) are not substitutes for `sdf_udf/` configs because 0107 explicitly requires matching `root_leaves`, `riggable_parts` and `root_leaves_ownership.json`.

## Original Oktopus references

The four known working reference SDF files were copied byte-for-byte into `sdf/`. The original puffer fused 0092 example was copied byte-for-byte into `sdf_udf/` as an archived-compatible reference.

## New paired experiments

For each of the 6 avatar/accessory combinations, there are three different configs:
1. `sdf/nov24_adapt_<pair>_sdf.yaml` — ordinary SDF candidate from the prior package. **No 0107 ownership assets.**
2. `sdf/paired_direct/nov24_adapt_<pair>_sdf_for_direct_0107.yaml` — *experimental* SDF with `root_leaves.parts` and `sdf_fused_exterior_0087: true`. Run this when intending to use direct cutter.
3. `sdf_udf/nov24_adapt_<pair>_sdf_udf_direct_0107.yaml` — paired *experimental* direct cutter YAML with byte-identical adaptation list and root leaf definitions. `tools/udf_run_direct.py` inserts the `udf_direct_fused_0100` selection at runtime.

Run `python verify_direct_pairs.py` to check that SDF and direct-UDF YAML adaptation lists and ownership are exactly matched. That test does NOT prove that the SDF mesher exports all required files or that UDF cutting produces correct geometry.

### Armadillo boots special case

The original armadillo boots SDF YAML has shaft and foot sharing `blend_group: boot` on **different avatar curves**. The 0107 cutter requires all owners of a part to share one target curve for its curve-interval assignment. Its paired-direct SDF *and* UDF YAMLs intentionally omit both `blend_group` declarations to make separate shaft and foot parts. **This changes geometry assembly** and must be visually evaluated. The ordinary `sdf/` YAML is left untouched.

### Other blockers and uncertainty

- These six new SDF mappings are starters, not validated adaptations.
- All-parts direct UDF thresholds were originally calibrated on Oktopus puffer body only. On new shapes, use `--allow-uncalibrated-nonbody` only for experimental previews; otherwise the 0107 code may refuse.
- The inspected 0107 runner demands `riggable_fused_sdf_exterior.ply`, `riggable_parts/<part>.ply`, and `root_leaves_ownership.json` from the **same SDF run**. If the current `tools/udf_run_direct.py` differs, check its `--help` and code.
- These paired-direct SDF YAMLs enable a fused 0087 extraction that may require disk-backed grid and substantial memory at high resolution; test at low resolution.

## Important compatibility: flat YAML basenames

The available 0107 runner has a check `Path(args.yaml).name == args.yaml` and rejects `--yaml sdf_udf/...`. You can make safe root-level **symlink aliases** without moving the structured files:

```bash
cd exp/train/manipulation
python make_flat_aliases.py --category sdf_udf       # dry-run
python make_flat_aliases.py --category sdf_udf --apply
```

Then use a basename (e.g. `--yaml nov24_adapt_trex_dress_sdf_udf_direct_0107.yaml`), which resolves through the symlink to `sdf_udf/`. To expose nested SDF basenames too, use `--category both`. This is not needed by runners that already accept relative subdirectory paths.

## Example: T-rex + dress direct-cutter experiment

Train joint T-rex+dress SDF and joint T-rex+dress UDF networks separately using your TWO training YAMLs in `config/`, then run SDF inference using the *paired direct* SDF YAML and check for all three 0107 required SDF artifacts. Finally:

```bash
python tools/udf_run_direct.py \
  --sdf-run YOUR_TREX_DRESS_PAIRED_SDF_RUN \
  --out trex_dress_udf_direct_preview \
  --shape trex_9_v1 \
  --config config/config_udf_grid_b13d3_trex1_dress2_v2.yaml \
  --yaml nov24_adapt_trex_dress_sdf_udf_direct_0107.yaml \
  --ckpt final --resolution 128 --parts all \
  --allow-uncalibrated-nonbody
```
This is an *experimental* command for 0107-style semantics. The UDF checkpoint must exist and the `--config` path must match how you trained.

## Migration safety

Extract into a *temporary* directory; review before copying the three folders to your repo. Do **not** delete or replace your current `exp/train/manipulation/` directory until you have confirmed SDF inference YAML resolution and checkpoint loading on your current branch.
