# 0101 — experimental direct-UDF unknown-cap preview (offline)

0100 separates **supported** (retain), **valid UDF absent** (remove), and **unknown** (retain). The 1024 puffer's bottom is split between absent and unknown. Removing **all** unknown would be unsafe: the body contains 285,664 unknown faces, and the body-only run marks every unqueried tentacle as unknown.

This script produces a **new preview**, never modifies 0100, 0081, 0092, SDF meshes, or inference. It does not need GPU or UDF evaluation. It proposes deleting only those UNKNOWN faces belonging to a **large, edge-connected, nearly planar, predominantly invalid-UDF component**, directly touching **independently UDF-confirmed removed faces** in the selected part. It reports all gates. This is a diagnostic heuristic, not proof that UDF predicts absence where its query was invalid. There is no silent fallback to deleting the whole part.

From Oktopus repository root (copy the script into `tools/`):

```bash
python tools/udf_0101_unknown_cap_preview.py \
  --direct inference/4M/puffer_udf_direct_fused_0100_r1024/oktopus_9_v1/udf_direct_fused_0100 \
  --part body \
  --out inference/4M/puffer_udf_0101_body_unknown_cap_trial
```

For a body-only 0100 experiment, `--direct` must refer to the body-only run. Inspect `report.json` and `body_unknown_cap_candidate.ply` first, then compare `fused_kept_cap_preview.ply` to `fused_direct_kept.ply`. Preview files appear only if a candidate passes all safeguards; if none pass, inspect the component audit instead of loosening thresholds indiscriminately.

The preview retains all tentacle faces, known-supported body faces, and unknown body faces outside the selected cap candidate. The output may be open or non-manifold; do not substitute it into production without inspection and additional geometry validations.

The generic resolution-independent correction remains to improve model query *validity* at caps, not declare invalid values to be distance-to-surface evidence. This utility tests whether a well-defined unqueried cap patch can be isolated *after* direct UDF inference while that correction is investigated.
