# 0103 — edge-connected unknown removal preview (offline, experimental)

This is a **diagnostic preview**, not a production cutter. It uses the saved 0100/0102 `consistency.npz` classification without rerunning SDF or learned UDF inference. It adds some `unknown` regions to the already confirmed `remove` mask, but **does not interpret all unknown queries as proof of missing surface**.

The existing 0101 preview required planarity; it selected the flat puffer bottom (component 3) but left a curved unknown region near the neck (component 14). 0103 does not use planarity or an absolute number-of-faces gate. Instead, an eligible unknown region must:

1. Be **edge-connected on the fused mesh within one routed root/leaf part** (not Euclidean nearest-neighbor proximity; no jumping to a folded tentacle turn).
2. Have at least 0.5% of its part's **physical triangle area** (not 8, 128 or any other fixed number of triangles).
3. Have >=60% invalid-UDF area within that connected region.
4. Have >=80% of its boundary contacts with **valid, confirmed UDF removals**, rather than known-supported faces; at least four such mesh-edge contacts are required.
5. Never include any known-supported face, and never cross a routed-part boundary.

These are conservative *preview gates*, not a mathematical test of a missing learned-UDF surface. In a Boolean-fused mesh that genuinely joins two touching sections, mesh adjacency alone may still connect an unintended route: inspect every non-body candidate carefully.

## Body first

Run from repository root, using the original 0100 body-only inference:

```bash
python tools/udf_0103_connected_unknown_preview.py \
  --direct inference/4M/puffer_udf_direct_fused_0100_r1024/oktopus_9_v1/udf_direct_fused_0100 \
  --parts body \
  --out inference/4M/puffer_udf_0103_body_connected_preview
```

The resulting `body_newly_selected_unknown.ply` and `fused_kept_connected_unknown_preview.ply` are the first visual checks. `report.json` audits each component. The result should be assessed against 0101: the large planar component and potentially the previously rejected curved component should become eligible **only if their boundary criteria are met**. If it still damages the neck/sidewall, do not enable it in production.

## All parts (only after running corrected 0102 and inspecting 0100 direct UDF masks)

```bash
python tools/udf_0103_connected_unknown_preview.py \
  --direct inference/4M/puffer_udf_direct_fused_0102_all_r1024/oktopus_9_v1/udf_direct_fused_0100 \
  --parts all --allow-uncalibrated-nonbody \
  --out inference/4M/puffer_udf_0103_all_connected_preview
```

Use your actual `0102` run name if different. The `--allow-uncalibrated-nonbody` flag is intentional: non-body 0100 thresholds have **not** been validated on curled tentacles. The report must be audited per part before accepting any candidate.

Outputs include `fused_kept_connected_unknown_preview.ply`, `fused_newly_selected_unknown.ply`, per-part selected unknown PLYs, `selection_0103.npz`, and `report.json`. Every output goes into a fresh folder. Original 0100, 0101, 0102, 0081, 0092 and SDF files are untouched.
