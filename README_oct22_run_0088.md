# 0088 — October 22 SDF / UDF run commands

**Requires** 0087 fused-SDF code to be installed in this Oktopus repository. All commands run from the repository root with the Oktopus conda environment, CUDA and DCSDD dependencies loaded. This bundle **does not** re-apply the 0087 source-code patch.

## Install this bundle

```bash
unzip -o ~/Downloads/oktopus-0088-oct22-configs-and-runners.zip -d .
```

This overwrites the October 22 YAML snapshots with the correctly updated puffer SDF/UDF pair and adds `tools/oct22_run.py`, `test_dc_oct22.sh`, `test_udf_oct22.sh` and offline tests. It does **not** overwrite your existing `test_dc_all.sh` or `test_sdf_udf_cut.sh`.

Before starting a long inference:

```bash
ls -l src/app/sdf_fused_report_0087.py
bash test_dc_oct22.sh --case oktopus_puffer_full --resolution 512 --require-fused --dry-run
```

## 1. Run the actual fused SDF first (puffer, mixed wrap + rigid)

```bash
bash test_dc_oct22.sh \
  --case oktopus_puffer_full \
  --shape oktopus_9_v1 \
  --resolution 512 \
  --require-fused \
  --out puffer_oct22_fused_sdf_r512
```

Defaults are `-a sdf_dc`, DCSDD `ours`, 5 outer / 3 inner iterations, `--sdf-localize-after-gate`, bounded filtering, fast bounding boxes, crop grid and disk-backed memmap under `logs`. If DCSDD is in a separate working tree, add:

```bash
--dcsdd-repo /fast/pselvaraju/Oktopus_now/third_party/dcsdd
```

**Source adaptation YAML**: `exp/train/manipulation/oct22_adapt_oktopus_puffer_full_sdf_optimization.yaml`.
It is the validated 0067 geometry plus `sdf_fused_exterior_0087: true`; all 15 adaptation records remain unchanged. The 0087 feature requires the source code patch, `combine_adaptations: true`, and the body + 7 leaves root/leaf configuration.

**Expected under** `inference/4M/puffer_oct22_fused_sdf_r512/oktopus_9_v1/`:

- `riggable_fused_sdf_exterior.ply`: **actual SDF min-union exterior**, not the unstitched assembly
- `riggable_fused_sdf_exterior_report.json`: counts, boundaries, manifold diagnostics
- `riggable_parts/`: body, T1, T2, T3, T4, T5, T6, T8, unchanged separate meshes
- `root_leaves_ownership.json`: ownership metadata for later UDF

Inspect the **fused exterior alone**, and check its report. The SDF union cannot remove a cap exposed on the outside of the union (e.g. T3's distal end); UDF opening cuts remain a separate stage.

1024 requires a **new output directory name** and extra scratch capacity:

```bash
bash test_dc_oct22.sh --case oktopus_puffer_full --resolution 1024 \
  --out puffer_oct22_fused_sdf_r1024 --require-fused
```

## 2. Other SDF cases: same `test_dc.py` mesher

**Single rigid boot**:

```bash
bash test_dc_oct22.sh --case oktopus_boots_t7_boot1_only \
  --resolution 512 --out boots_t7_oct22_sdf_r512
```

**Two wrap adaptations, combined at the SDF stage (cone)**:

```bash
bash test_dc_oct22.sh --case oktopus_cone_post_sdf \
  --resolution 512 --out cone_oct22_sdf_r512
```

**Other mixed adaptations**, e.g. a mixed armadillo YAML, can be used for SDF, but must specify their matching model config and shape:

```bash
bash test_dc_oct22.sh --case armadillo_boots2 --shape armadillo_13_v1 \
  --config config/<correct_armadillo_model_config>.yaml \
  --resolution 512 --out armadillo_boots2_oct22_sdf_r512
```

The `<correct_armadillo_model_config>` above is a placeholder: use the model config actually used for that dataset. Do **not** simply copy the Oktopus puffer model config. Some archived experiment YAMLs have radius flags missing/ambiguous; the runner rejects such configurations until the source settings are resolved.

### Radius/merging rules

| Geometry | Per-adaptation YAML | Merge setting | Result |
| --- | --- | --- | --- |
| Independent rigid | `rigid_radius: true`, `wrap_radius: false` | default/no `combine_adaptations` | Individual adapted mesh |
| Independent wrap | `wrap_radius: true`, `rigid_radius: false` | default/no `combine_adaptations` | Individual adapted mesh |
| Two overlapping wraps | two wrap records | `combine_adaptations: true`, optional `blend_groups` | Combined/blended SDF |
| Mixed root+leaves puffer | 1 wrap body, 14 rigid sections | `combine_adaptations: true`, `root_leaves`, `sdf_fused_exterior_0087: true` | Eight riggable parts **and** true Boolean-unioned SDF exterior |
| Mixed with no merge specification | rigid+wrap records | not enabled | No guaranteed merged exterior; runner warns |

Do not use `sdf_fused_exterior_0087` on arbitrary boots/dress/cone configurations; the current 0087 implementation is only integrated with the full-puffer sparse root/leaf path. **Radius selection is inside each adaptation, not a CLI flag.**

## 3. Optional existing UDF-on-eight-parts command (NOT UDF on fused exterior)

Once you have examined the SDF mesh:

```bash
bash test_udf_oct22.sh --case oktopus_puffer_full \
  --sdf-run puffer_oct22_fused_sdf_r512 \
  --resolution 512 --out puffer_oct22_udf_parts_r512
```

This invokes the existing `test_udf_sdf_cut_stitch.py` and `oct22_adapt_oktopus_puffer_full_udf_optimization.yaml`, overriding the **riggable_parts** source path to the named SDF run. It saves the eight cut meshes and attempts 0084 boundary stitching. That `STITCHED.ply` is **not Boolean-unioned**; it may contain internal overlapping geometry.

**Not yet implemented by 0084:** UDF cutting directly on `riggable_fused_sdf_exterior.ply`, producing one cleaned fused final exterior. That needs an additional fused-surface owner/curve mapping. **Do not treat 0084's stitched PLY as that output.**

## 4. Import troubleshooting

If the UDF runner still raises `ModuleNotFoundError` even though the file exists:

```bash
python -c 'import tools; print(tools.__file__, list(tools.__path__)); from tools.promote_udf_0081_assembly import promote; from tools.udf_stitch_0084 import stitch_parts; print("imports OK")'
```

The script performs this same import check before starting UDF inference. The directory `/Oktopus/tools/` must be imported from the repository, not from a third-party `tools` package.

## Offline validation

```bash
python -m unittest discover -s tests -p test_oct22_runner_0088.py -v
```

SDF and UDF model execution and real puffer geometry have **not** been run in this sandbox.
