# 0107 — SDF → direct UDF → connected cleanup (no previous UDF run)

0107 removes the *saved 0092 UDF cut evidence* dependency from the 0100/0106
direct-UDF workflow. It does not change the learned UDF, surface classification,
0103 connected UNKNOWN proposal gates, or largest-edge-component cleanup.

## Inputs: one completed SDF run

The selected SDF shape directory must contain:

- `riggable_fused_sdf_exterior.ply` — the Boolean-fused SDF exterior
- `riggable_parts/<part>.ply` — the SDF part meshes
- `root_leaves_ownership.json` — manifest with part names, adaptation fingerprint,
  and item-to-part mapping (validated against the UDF YAML)

**No 0081 run, no 0092 report, no `fused_udf_transfer_evidence_0092.npz`,
and no `--udf-run` are needed.**

## One UDF command after SDF

From the Oktopus repository root:

```bash
python tools/udf_0100_run_direct.py \
  --sdf-run adapt_oktopus_puffer_sdf_r1024 \
  --out puffer_udf_0107_from_sdf_r1024 \
  --resolution 1024 \
  --parts all \
  --allow-uncalibrated-nonbody \
  --ckpt final
```

`--allow-uncalibrated-nonbody` remains **experimental** because the tentacle
UDF thresholds have not been independently calibrated. The script also accepts
`--parts body` without that flag. `--udf-run` is accepted for backwards
compatibility but **ignored**; it will not be used as input.

The script (a) loads the SDF fused mesh and SDF parts, (b) routes fused faces
by nearest **aligned triangle surface** under a distance tolerance, (c) queries
UDF directly on the fused faces with their adaptation owners, and (d) calls
0103 connected-unknown removal and largest-edge-component cleanup internally.

### Outputs

```
inference/4M/puffer_udf_0107_from_sdf_r1024/oktopus_9_v1/
  udf_direct_fused_0100/
    consistency.npz          # direct supported / absent / unknown masks
    report.json              # includes routing_0107
    fused_direct_removed.ply
    fused_direct_unknown.ply
  udf_connected_final_0106/
    fused_kept_connected_unknown_preview.ply   # cleaned final preview
    report.json              # includes cleanup_largest_component
```

The 0107 routing report has matched/unmatched face counts, per-part counts,
world-space distance tolerance and normal threshold. Faces without a valid
SDF-part match remain UNKNOWN, never automatically removed. Routing **aborts**
if more than 5% of fused faces cannot be matched; missing or mismatched SDF
ownership manifest and part meshes also abort before classification.

A 0092 transfer report is neither read nor generated. SDF Boolean seams may
still have small unmatched regions. Inspect `routing_0107` and inspect
`fused_kept_connected_unknown_preview.ply` on real shapes, especially tight
folded tentacles and neck/bottom opening boundaries. This remains experimental
until the all-part UDF thresholds and GPU/mesh results are validated.

## Quick checks

```bash
python -m pytest -q tests/test_udf_fused_part_ownership_0107.py \
  tests/test_udf_0102_all_parts_cli.py

# Does not launch inference:
python tools/udf_0100_run_direct.py \
  --sdf-run adapt_oktopus_puffer_sdf_r1024 \
  --out preview_0107_dryrun \
  --parts all --allow-uncalibrated-nonbody --dry-run
```

**Note:** This change eliminates prior UDF **file dependency**, not the need
to load the UDF model/checkpoint and evaluate its field. Neither the raw
UDF model nor its existing geometry caveats are altered.
