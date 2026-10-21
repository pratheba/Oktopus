# 0072: Recover the October 18 UDF method, transfer onto 0067 SDF final meshes

This patch restores the **original Git-history implementation** removed by Oct 19
commit `21174f9` (parent commit `25e4e19`):
- `_absolute_hysteresis_with_saturation_guard`, including no-cut far plateau
- `_estimate_udf_metric_calibration` (finite-difference gradient-norm metric)
- Per-adaptation tiny-fragment cleanup, with fragments included in removed evidence
- Legacy removed/kept nearest-surface transfer already in the existing code

The new YAML switches from 0066–0071 direct-final UDF to the previous `transfer`
mode, with 0067's **eight final meshes as geometry authority**.

This is a **recovery run**, not yet a mesh-reference-free generalized method.
It re-evaluates UDF on the original 15 saved SDF per-adaptation PLYs. It does NOT
rerun or modify SDF/0067, and does NOT use the old, mixed-timestamp NPZ masks.

## Install (on top of 0071)

```
cd /lustre/fast/fast/pselvaraju/Oktopus_now
git apply --check ~/Downloads/oktopus-0072-restore-legacy-metric-transfer.patch
git apply ~/Downloads/oktopus-0072-restore-legacy-metric-transfer.patch
```

## Check original 15 individual input meshes exist

```
python - <<'PY'
from pathlib import Path
p = Path('inference/4M/adapt_oktopus_puffer_optim_512/oktopus_9_v1')
names = ['body'] + ['l_arm']*3 + ['r_arm']*11
missing = [str(p / f'{i}_direct_puffer_3_v1_{name}.ply')
           for i, name in enumerate(names)
           if not (p / f'{i}_direct_puffer_3_v1_{name}.ply').is_file()]
print('Missing reference meshes:', missing)
PY
```

If the original individual meshes are not available but you have the provided
`test_adapt_oktopus_puffer_udf_localize10_mc512.zip`, extract its 15
`*_mc_before_udf_cut.ply` files and use the alternate `--mc-mesh` pattern
pointing to those files. The Oct 18 JSON reports and Oct 17 NPZ files **are
from different runs**; do not replay the NPZ masks as if from Oct 18.

## Run UDF only

```
python test_sdf_udf_cut.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o puffer_0072_legacy_metric_transfer_r512 \
  -s oktopus_9_v1 \
  -y udf_adapt_oktopus_puffer_0072_verified_transfer.yaml \
  -r 512 -ckpt final \
  --mc-mesh 'inference/4M/adapt_oktopus_puffer_optim_512/oktopus_9_v1/{index}_{mode}_{accessory}.ply' \
  --udf-far-value 0.1 \
  --udf-model-batch-size 32768 \
  --udf-cut-query-batch-size 32768
```

`--mc-mesh` points to the 15 **reference PLY inputs** (MC or DCSDD both work).
The YAML directs the final output to the 8 existing `puffer_0067_final_parts_r512`
meshes. Do not add flags to enable the 0069–0071 direct/cap heuristics.

## Verify

```
grep -E '^\\[udf cut (metric threshold|fragment cleanup|root_leaves)\\]' run.log
```

You should see per-adaptation calibrated raw thresholds (not global 0.02/0.01)
followed by per-part transfer counts. Inspect cut meshes for unwanted tentacle
openings; this changes reference sources and final output geometry compared
with the old run, so visual agreement is **not guaranteed**.

## Test status

- 25 targeted tests PASS in a local copy, including 4 new 0072 tests.
- Git patch application and Python syntax verified on 0071 tree.
- The real model + eight resulting meshes could not be run in this environment.
