Oktopus 0052 — staged puffer candidate ROI + body-only benchmark
================================================================

Apply ON TOP OF 0051a. Patch and replacement ZIP are alternatives.

Scope
-----
1. The 0048a tangent-aware source-gate envelope may now be used by the
   exact 0051a bounded, correction-rich puffer minimal staging path.
   Previously 0048a rejected anything without a streaming callback.
2. Audited ROI ensures any excluded candidate that passes the original
   localizer aborts BEFORE neural inference. A sampled spatial envelope is
   not an analytic proof: run the audit for every changed setting/geometry.
3. Optional sdf_puffer_stop_after: 1 runs the body, exports its individual
   DCSDD mesh, and exits. This is a PARTIAL BENCHMARK, NOT a puffer result.
4. The existing full per-section global support-limit quantiles, avatar
   cutting, detail recomposition, seam optimization, and UDF ownership
   remain unchanged. This patch does NOT eliminate inference-data staging,
   enable puffer callback inference, or stitch body and tentacle geometry.

Install
-------
  cd /fast/pselvaraju/Oktopus_now
  unzip -o ~/Downloads/oktopus-0052-puffer-staged-roi-files.zip

For the initial 512 body audit run, reuse your WORKING 0051a command but use
  -y adapt_oktopus_puffer_root_leaves_0052_roi_body_audit.yaml
  -o test_adapt_oktopus_puffer_0052_body_audit_r512
and retain your --dcsdd-methods ours, --dcsdd-bounded-filtering,
--dcsdd-crop-grid, --sdf-localize-after-gate, memmap dir, etc.

IMPORTANT: -y searches under exp/train/manipulation/ in the user's setup.
If needed, pass its full path.

Expect:
  [sdf_puffer_0052] PARTIAL BENCHMARK ONLY: stop_after= 1 of 15 ...
  [sdf_puffer_0052] candidate_roi=staged_puffer mode= audit ...
  [sdf_candidate_interval_0048 audit] ... missing_accepted= 0 ...
  [filter_grid_adapt] ... candidate_samples= ... after_localize= 15466060
  [sdf_puffer_0051] minimal_staging=True ...
  [sdf_puffer_0052] PARTIAL BENCHMARK COMPLETED ...

Baseline from user's 0051a body:
  candidate_samples=120794571
  after_localize=15466060
  localize_compute_s=210.113
  stage_write_s=117.562
  staged_GB=2.351
  final crop=213^3, DCSDD active cells=189928, edges=191174.

The audit deliberately adds time. Compare accepted samples and SDF/DCSDD
statistics. If missing_accepted > 0 the run raises ValueError UNSAFE:
turn pruning OFF, and DO NOT try unaudited production.

ONLY AFTER a successful audit for the exact same 512 inputs, run
  -y adapt_oktopus_puffer_root_leaves_0052_roi_body_AFTER_AUDIT_ONLY.yaml
  -o test_adapt_oktopus_puffer_0052_body_fast_r512

Do NOT extrapolate this sampled-envelope audit to 1024 or 2048. Repeat
there and add margin or disable pruning if audit fails.

To test rigid sections next, edit sdf_puffer_stop_after: 4; keep
sdf_candidate_interval_audit: true until all four sections pass.

Next engineering step
---------------------
Direct streaming with correction parity requires at least a two-stage
calibration/replay or equivalent approach because clamp_pred_sdf_by_support
computes per-curve quantiles using the FULL model prediction arrays.
Calling it separately per 262144-point chunk changes geometry.

Verification
------------
39 targeted unit tests passed (new staged rigid/wrap ROI and audit fail-closed,
0048 candidate audit, 0051 staged parity, root-leaf and sparse group tests).
No neural puffer benchmark was executed in this container.
