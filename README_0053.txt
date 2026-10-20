Oktopus 0053 — bounded direct puffer predictions, preserved whole-section clamp
============================================================================

APPLY ON TOP OF 0052 (which itself requires 0051a/0050a/0050/0049).
The files ZIP and Git patch are alternatives: use one, not both.

Scope:
  - Opt-in sdf_puffer_direct_predictions_0053: true. This ENABLES one pass
    over candidate voxels: localize accepted samples -> buffer up to one
    accepted inference batch -> predict accessory and avatar -> append compact
    predictions, source/target coordinates and support inputs to disk.
  - The existing full-section, global rho-quantile support clamp (both final
    and base) runs *after* all predicted rows have been collected, exactly
    as the old correction code did. No per-batch support quantiles.
  - Original avatar cut, base snug/offset, detail ownership/gating, group
    composition, DCSDD extractor and root/leaf ownership remain unchanged.
  - Existing 0048a conservative candidate pruning and audit still apply.
  - Unsupported auto-avatar snug field, tiled detail and curvature-based final
    carving FAIL CLOSED before localization rather than changing semantics.

IMPORTANT LIMITATIONS:
  - This is NOT a fully online final-SDF writer: correction still needs all
    accepted predictions for global calibration and replays them afterward.
  - Predictions and small calibration features are stored in scratch memmaps;
    avoid sharing logs/scratch across overlapping runs.
  - Final SDF processing still uses whole-section NumPy arrays. The current
    implementation is NOT guaranteed to fit in memory or run fast at 2048.
  - Model calls are grouped by accepted rows (up to 262144, not candidate
    batches). This differs slightly from global np.array_split grouping.
    Floating-point mesh equality requires a measured real-model parity check.
  - No root/leaf welding, UDF change or animation.
  - A sampled candidate envelope is not a proof for higher resolution.

Install:
  cd /fast/pselvaraju/Oktopus_now
  unzip -o ~/Downloads/oktopus-0053-puffer-direct-predictions-files.zip

First run: reuse the known working 512 command, change only:
  -y adapt_oktopus_puffer_root_leaves_0053_body_audit.yaml
  -o test_adapt_oktopus_puffer_0053_body_audit_r512
Keep --dcsdd-bounded-filtering, --dcsdd-crop-grid, --surface-extraction ours,
--sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs and other DCSDD flags.
Do NOT add the independent Boot streaming flags or --sdf-stream-unified.

Look for:
  [sdf_puffer_0052] candidate_roi=staged_puffer mode=audit
  [sdf_candidate_interval_0048 audit] ... missing_accepted=0
  [sdf_puffer_0053] predicted_rows=15466060 ... global_support_clamp=deferred_exact
  [sdf_localize_0041] ... localize_s=... inference_write_callback_s=...
  [dcsdd:grid] evaluated=5021570 ... active cells=189928 ...
  [sdf_puffer_0052] PARTIAL BENCHMARK COMPLETED adaptations=1 ...

Body-only baseline at 512 from 0052:
  candidates=108190890 accepted=15466060
  localization_compute_s=195.260, stage_write_s=26.279, staged_GB=2.351
  filter_s=252.058, end-to-end=363.857 (INCLUDING audit 23.557 s)
  SDF crop=213^3; evaluated=5021570; negative=1381111
  sign-changing edges=191174; active cells=189928
  body mesh watertight=False; support-boundary active cells=40608

Compare accepted count, SDF distributions/negative/evaluated, sign edges,
active cells, mesh extent and saved PLY. Differences? Keep 0052 as the
reference; do not scale up. A mere matching count is insufficient for strict
field parity — use the 512 output and compare per-vertex/field data if available.

Only after passing the body audit and geometry comparison:
  -y adapt_oktopus_puffer_root_leaves_0053_body_after_audit.yaml
Then test through t1 (4 adaptations) WITH candidate audits:
  -y adapt_oktopus_puffer_root_leaves_0053_t1_audit.yaml

Tests: synthetic callback parity for rigid/wrap; exact accepted index,
model inputs, global support quantiles; fail-closed flags; existing puffer
bounded, minimal staging, 0052 ROI, seam, sparse group and ownership tests.
No actual puffer GPU run possible in this environment. Measure speed and
scratch size on cluster; do not claim speedup before benchmarks.
