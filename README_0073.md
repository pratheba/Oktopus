# Oktopus 0073: recover legacy metric calibration in direct-final UDF mode

## Scope

Use the existing 0067 SDF final eight `riggable_parts` meshes. No `--mc-mesh`, no SDF rerun, no intermediate reference meshes.

0073 restores the exact legacy October 18 **absolute-hysteresis + world-space UDF gradient calibration** to selected owners in the direct-final cutter:

- `udf_cut_direct_metric_parts_0073`: parts whose owners get per-owner metric calibration. Body stays on the existing direct-final behavior.
- `udf_cut_direct_dry_run_parts_0073`: run the metric-based cut, export proposed removed patches but preserve original mesh geometry for these parts.
- `udf_cut_direct_export_removed_0070: true` saves each proposed cut as `udf_cut_removed_diagnostics_0070/<part>_removed.ply`.
- `udf_cut_metric_calibration_samples`, `udf_cut_metric_calibration_min_samples`, `udf_cut_metric_calibration_epsilon_edges`: recovered legacy calibration controls.

New log lines:

```
[udf cut metric 0073] item=... status=ok scale=... samples=... seed=... grow=...
[udf cut dry run 0073] part=tentacle_t... candidate_removed=... action=preserve_input
```

If a UDF owner's data are saturated or calibration fails, its contribution abstains instead of using uncalibrated .02/.01 thresholds.

**Critical limitation:** Sampling and independent removed/kept reference meshes differ from the old pipeline. 0073 recovers the *metric calibration formula* but **cannot guarantee reproduction of the legacy reference-transfer cuts**. The first run is diagnostic and cannot damage saved SDF output.

## Install

```
cd /lustre/fast/fast/pselvaraju/Oktopus_now
git apply --check ~/Downloads/oktopus-0073-direct-final-metric-calibration.patch
git apply ~/Downloads/oktopus-0073-direct-final-metric-calibration.patch
```

## First experiment: diagnostic, no tentacle cuts applied

```
python test_sdf_udf_cut.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o puffer_0073_metric_diagnostic_r512 \
  -s oktopus_9_v1 \
  -y udf_adapt_oktopus_puffer_0073_metric_diagnostic.yaml \
  -r 512 -ckpt final \
  --udf-far-value 0.1 \
  --udf-model-batch-size 32768 \
  --udf-cut-query-batch-size 32768
```

Check diagnostics:

```
grep -E '^\[udf cut (metric 0073|dry run 0073|final part|final root_leaves)' run.log
```

Inspect `inference/4M/puffer_0073_metric_diagnostic_r512/oktopus_9_v1/udf_cut_removed_diagnostics_0070/tentacle_t*_removed.ply` for **where** the new decisions would cut.

## Only after inspecting proposals

`udf_adapt_oktopus_puffer_0073_metric_apply.yaml` applies the proposed tentacle cuts. It does **not** run automatically. Use a different `-o` output name. Do not proceed to boots/cone based only on numerical face counts.

## Validation

`python -m pytest -q tests/test_udf_cut_metric_direct_0073.py tests/test_udf_cut_absolute_metric_0072.py tests/test_udf_cut_final_root_leaves_0066.py tests/test_udf_cut_p95_cap_guard_0071.py tests/test_udf_cut_combined_direct.py`

26 tests passed in the source snapshot from `Oktopus(20261021-172243).zip`. Full inference against model checkpoints was not run locally.
