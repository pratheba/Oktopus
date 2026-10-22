# 0081 puffer: working 512 versus failing 1024 (read-only)

Goal: find where the earlier 512 learned-UDF cutting behavior changed. **This is a diagnostic, not a new cap-removal heuristic.** It does not alter source meshes, SDF generation, UDF inference, the 0081 selection algorithm, or 0092 fusion.

## What the repository confirms

- Original 0081 (`udf_adapt_oktopus_puffer_0081_curve_local_preview.yaml`) and 0092 UDF YAMLs have identical 15 `adaptations` entries.
- Their four 0081 locality settings match: 0.11 curve band, 0.07 arclength neighbor step, 0.40 surface walk, 8 min seed faces.
- The old 0081 YAML takes `puffer_0067_final_parts_r512/oktopus_9_v1/riggable_parts`; the current 0092 run uses the SDF run selected by `SDF_RUN` in the editable shell runner. `tools/oct22_run.py` writes a temporary runtime YAML with the requested input mesh directory.
- 0092 transfers already saved UDF selections and does not recompute the UDF.
- Successful earlier 0081 preview was named `puffer_0081_curve_local_preview_r512` in README_0081.md; its output at `.../udf_cut_curve_local_0081/` must be on disk for direct comparison.

## Cluster commands

```bash
cd /fast/pselvaraju/Oktopus_now
python tools/udf_0081_compare_runs.py --discover inference/4M

python tools/udf_0081_compare_runs.py \
  --old inference/4M/puffer_0081_curve_local_preview_r512 \
  --new inference/4M/adapt_oktopus_puffer_udf_r1024 \
  --out /tmp/puffer_0081_512_vs_1024.json | tee /tmp/puffer_0081_512_vs_1024.txt
```

The tool finds both historic `run/shape/udf_cut_curve_local_0081` and newer `run/_udf_inference_0084/shape/udf_cut_curve_local_0081` layouts. If the working run was renamed or has been moved, use the actual directory printed by `--discover`.

The text report compares per-part face counts, seed/selection ratios, terminal-cap evidence, and the body's 20 arclength bins. The JSON contains the same details plus sampled, **approximate**, nearest-centroid geometry comparisons. Face counts alone are not comparable across resolutions; compare percentages, seed spans and bounding boxes.

## Reading the comparison

- **Working 512 has high-cap seeds, failing 1024 does not:** investigate changed per-face learned UDF evidence and source-mesh geometry, not just the 0081 growth radius.
- **Both runs have similar high-cap seeds but selection differs:** investigate graph connectivity, arclength projection and component growth.
- **Geometry bounds / curve lengths differ significantly:** investigate 0067 versus current source SDF reconstruction, vertex coordinates and curve mapping.
- **0081 saved cut correct but final fused PLY wrong:** investigate 0092 evidence transfer and fusion separately.
- **Working 512 locality.npz unavailable:** we cannot prove where the regression originated; do not apply a new end-cap heuristic just to mask the missing comparison.

## Existing failing 1024 evidence (uploaded)

`body/locality.npz`: 1,209,688 faces; 196,519 seeds; 539,461 candidates; 260,976 selected; 278,485 blocked. At high arclength s >= 0.89: 278,359 candidates, **0 seeds, 0 selected, 278,359 blocked**. Highest seed s = 0.763617. Other tentacle end candidates may correctly remain blocked, so **do not widen 0081 globally** based on the body result.
