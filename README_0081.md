# 0081 — Curve-local, surface-connected four-class UDF cutting preview

## Why

0080 successfully opened most puffer body and tentacle closures by taking
`{unassigned, no_raw_udf, seed_not_removed_other, agreed_removed}`. However,
those four classes are **evidence categories**, not anatomical ownership.
A sharply bent target curve may pass near a separate mesh surface in 3D.
Removing every unsupported face can carve legitimate sidewalls.

0081 uses **0079 independently agreed removed faces as trusted seeds** and
considers the other three 0080 classes as *growth candidates*. Each seed
component recruits candidates only if (1) a chain of shared mesh edges reaches
it; (2) each edge step has continuous target-curve arclength; (3) the candidate
stays within a padded arclength interval of that seed component; and
(4) the *surface-walk* distance is bounded. Projected curve arclength is
normalized to [0,1], not raw `src_0/src_1` parameter or nearest XYZ alone.
No body/tentacle IDs, coordinates, manually selected cut strips, or hardcoded
closure shapes are used.

**This is NOT a validated production algorithm.** Self-near curve projections
may still be ambiguous, and a false-positive *seed* can still be removed.
Inspect ALL eight proposed PLYs, especially body neck/bottom and T3/T6/T8.
Changing projection locality does not by itself prove correct anatomy.

## Apply on TOP of 0079

```bash
cd /lustre/fast/fast/pselvaraju/Oktopus_now
git apply --check ~/Downloads/oktopus-0081-curve-local-four-class.patch
git apply ~/Downloads/oktopus-0081-curve-local-four-class.patch
```

## Run once, UDF only

The 0081 YAML is installed to `exp/train/manipulation/`. It has the exact
original 15 adaptation definitions and points to the existing 0067 SDF
`riggable_parts/` directory.

```bash
python test_sdf_udf_cut.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o puffer_0081_curve_local_preview_r512 \
  -s oktopus_9_v1 \
  -y udf_adapt_oktopus_puffer_0081_curve_local_preview.yaml \
  -r 512 -ckpt final \
  --udf-far-value 0.1 \
  --udf-model-batch-size 32768 \
  --udf-cut-query-batch-size 32768
```

**No `--mc-mesh`, no SDF rerun, no avatar Boolean.**
The combined `riggable_after_udf_cut.glb` remains deliberately **uncut**.
The proposed outputs are in:

```
inference/4M/puffer_0081_curve_local_preview_r512/oktopus_9_v1/
  udf_cut_curve_local_0081/
    body/{proposed_kept_cleaned.ply, proposed_removed.ply,
          added_unsupported_faces.ply, blocked_candidate_faces.ply,
          locality.npz, summary.json}
    tentacle_t1/... through tentacle_t8/...
```

Missing empty PLYs are expected. `detached_fragments.ply` is emitted when
there are detached faces. The geometry cleanup keeps only the main
edge-connected component when it is at least 85% of the cut mesh; otherwise
it saves the uncleaned preview and reports the reason.

```bash
grep '^\[udf cut curve local 0081\]' run.log
```

## Offline tuning / no more UDF inference

Once the 0081 run has exported `locality.npz`, vary the arclength and
surface-distance controls without rerunning the UDF model:

```bash
python tools/udf_0081_replay_locality.py \
  --mesh-dir inference/4M/puffer_0067_final_parts_r512/oktopus_9_v1/riggable_parts \
  --run-dir inference/4M/puffer_0081_curve_local_preview_r512/oktopus_9_v1 \
  --out /tmp/puffer_0081_replay \
  --curve-band 0.11 --curve-step 0.07 --surface-walk 0.40
```

The offline tool verifies face centroids against the original mesh, refuses
mismatched input face ordering, and never overwrites input meshes.
`--curve-band`, `--curve-step`, `--surface-walk` are **fractions** of the
sampled curve length (except `--curve-step`, normalized delta of an edge).
`--min-seed-faces` defaults to 8.

## Caveats / acceptance

- These locality limits are initial **experimental values**, not trained or
  validated puffer/boot/cone defaults.
- A component with strong but **incorrect** 0079 remove votes remains a seed;
  it is not automatically healed by locality.
- Legitimate large body openings may be **under-cut** relative to the successful
  0076/0078 four-class body reference. Compare visually. Do not copy the
  earlier four-class rule to all tentacles or apply this patch directly.
- Check the tentacle curl region nearest the body and the body neck/bottom.
- No smoothing, boundary-hole filling, or production-GLB cutting in 0081.
- The only code path affected is `udf_cut_curve_local_preview_0081: true`.

Tests:

```bash
PYTHONPATH=src:src/app python -m pytest -q \
  tests/test_udf_cut_curve_local_0081.py tests/test_udf_cut_reconcile_0079.py \
  tests/test_udf_cut_final_root_leaves_0066.py tests/test_udf_cut_combined_direct.py
```
