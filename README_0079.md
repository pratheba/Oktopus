# 0079 — per-adaptation evidence reconciliation on 8 final root/leaf meshes

**This is an experimental evidence-based UDF cutting *preview*, not a validated final cutter.**

## Problem addressed

0069 assigned each final tentacle face to a *single* adaptation, even where
multiple `src_0/src_1` intervals overlap. Thus a neighboring learned UDF never
had a chance to protect a blended sidewall. 0079 evaluates every eligible
owner within the curve intervals, runs independent metric-calibrated UDF
seed/grow component tests, and keeps the face unless **all assigned owners
provide valid, accepted remove evidence**. Unknown/missing UDF evidence is
kept, never reinterpreted as a cap. The 15 adaptations remain logical evidence
records; no 15 separate DCSDD meshes and no avatar Boolean are required.

The opt-in mode exports: per-part `votes.npz` (assigned, valid, remove and keep
votes, missing/conflict masks); `pre_topology.ply`, `final_removed.ply`,
`proposed_kept_cleaned.ply`, and optionally `detached_removed.ply`. Generic
largest-edge-connected-component cleanup is applied to the *preview* only.
Connected rim spikes and holes are NOT smoothed/fixed by this patch.

**Important limitation**: the 0076 body result intentionally removed
`no_raw_udf` faces to open the neck and bottom. That cannot safely be made
universal: 0074 showed missing support also on several tentacles. 0079
therefore retains unknown faces. The new preview may still leave the body's
neck/bottom closed. It is a prerequisite evidence-reconciliation experiment,
not a claim that all eight openings are now recovered. The exact old
15-reference removed/kept transfer is still not reproducible from raw final
meshes without assumptions.

## Installation

Applies **on top of 0074**, not just the October 21 pre-0074 uploaded source.

```bash
cd /lustre/fast/fast/pselvaraju/Oktopus_now
git apply --check ~/Downloads/oktopus-0079-udf-owner-reconcile.patch
git apply ~/Downloads/oktopus-0079-udf-owner-reconcile.patch
```

## 512 preview run (SDF already complete)

```bash
python test_sdf_udf_cut.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o puffer_0079_reconcile_preview_r512 \
  -s oktopus_9_v1 \
  -y udf_adapt_oktopus_puffer_0079_reconcile_preview.yaml \
  -r 512 -ckpt final \
  --udf-far-value 0.1 \
  --udf-model-batch-size 32768 \
  --udf-cut-query-batch-size 32768
```

The paired UDF YAML refers to the **existing** `puffer_0067_final_parts_r512`
`riggable_parts` and has exactly the same 15 adaptation definitions as the
original SDF YAML. No `--mc-mesh` argument, no SDF rerun.

Outputs under:
`inference/4M/puffer_0079_reconcile_preview_r512/oktopus_9_v1/udf_cut_reconcile_0079/<part>/`

The resulting GLB **preserves all eight input meshes intentionally**. Inspect
`proposed_kept_cleaned.ply` for each part, and `votes.npz` for voting causes.
The `udf_cut_combined_direct_preserve_components: false` setting is for
non-destructive preview only. **Do not use these settings for production cuts.**

## Logs and validation

```bash
grep -E '^\[udf cut (owner intervals 0069|metric 0073|reconcile 0079|cleanup 0079|dry run 0073)' run.log
```

```bash
python -m pytest -q tests/test_udf_cut_reconcile_0079.py tests/test_udf_cut_evidence_0074.py tests/test_udf_cut_final_root_leaves_0066.py tests/test_udf_cut_combined_direct.py
```

The reconcile mode is opt-in, so earlier pipelines retain their behavior.
There is an explicit apply authorization guard to prevent accidental use of
experimental votes for modifying final outputs.
