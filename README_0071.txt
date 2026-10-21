Oktopus 0071 — Tentacle UDF: final mesh, conservative endcap check
==================================================================

Status: EXPERIMENTAL, NOT YET VERIFIED ON PUFFER MESHES.

Requires 0066 corrected runner, 0069 curve ownership, and 0070 safety patch.
The patch is incremental and applies only to the existing 0070 files.

Why:
  0069 uses a high UDF score to remove connected components on final blended
  tentacle surfaces. Blending and seams can make legitimate sidewalls appear
  far from the independent learned UDF. 0069 caused unwanted holes. The 0070
  body-only fallback prevented holes but did not recover tentacle openings.

0071 keeps final root/leaf DCSDD meshes and adds:
  (1) Optional P95 per-owner calibration excluding UDF's far-value shoulder,
      based on the original auto P95 FORMULA (not the original reference mesh).
  (2) Conservative geometry gate: accept a UDF grow-component only when most
      of its faces are close to a selected terminal source-curve interval and
      have normals parallel to the centerline direction (cap-like).
      This rejects sidewall stripes rather than clipping arbitrary patches.
  (3) Diagnostic blocked-face PLY exports; original body path is unchanged.

IMPORTANT limitations:
  * The original UDF reference-transfer cut is NOT fully restored. In the
    original, per-item SDF meshes supplied a removed-versus-kept reference;
    those meshes were not exported in the 0067 SDF run.
  * The 'low' curve end is assumed to be the attachment/root end. If this is
    false for a tentacle, it may reject its intended cut or accept an
    unintended cut. Compare the cut PLYs against the original SDF geometry.
  * Even a planar end cap at the 'low' end might be intended to remain closed.
    UDF score is still required; geometry cannot prove intent.
  * Do not treat P95 as equivalent to the fixed 0.02/0.01 tuned thresholds.
  * No access to the actual 0067 puffer meshes or trained UDF model here;
    only synthetic geometry/unit tests were run.

Install:
  cd /lustre/fast/fast/pselvaraju/Oktopus_now
  git apply --check ~/Downloads/oktopus-0071-udf-cap-guard.patch
  git apply ~/Downloads/oktopus-0071-udf-cap-guard.patch

No rerun of test_dc.py. Both YAMLs point to the existing:
  inference/4M/puffer_0067_final_parts_r512/oktopus_9_v1/riggable_parts

Trial A — fixed tuned thresholds, new end-cap gate:
  python test_sdf_udf_cut.py \
    -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
    -o puffer_0071_guard_fixed_r512 \
    -s oktopus_9_v1 \
    -y udf_adapt_oktopus_puffer_0071_guard_fixed.yaml \
    -r 512 -ckpt final \
    --udf-far-value 0.1 \
    --udf-model-batch-size 32768 \
    --udf-cut-query-batch-size 32768

If trial A rejects required openings, compare trial B below:
  same command but:
    -o puffer_0071_guard_p95_r512 \
    -y udf_adapt_oktopus_puffer_0071_guard_p95.yaml

Diagnostics:
  grep -E '^\[udf cut (owner intervals 0069|combined direct owner|p95 0071|cap guard 0071|blocked 0071|final part)' run.log

Outputs include the same eight final cut PLYs and GLB as 0069, plus:
  udf_cut_removed_diagnostics_0070/<part>_removed.ply
  udf_cut_blocked_diagnostics_0071/<part>_blocked.ply
and JSON report:
  riggable_after_udf_cut_report.json
  .parts.<part>.direct_udf.cap_guard_0071
  .parts.<part>.direct_udf.owners[].calibration_0071

0071 is opt-in: no keys -> 0070 and 0069 execution unchanged.
The adaptation list is identical to 0070/0067 for manifest fingerprinting.

Validation in this container:
  21 targeted tests passed: 0071 cap guard, calibration, 0066/0069 direct UDF.
  Broader pre-existing test suite has missing historical source methods and
  fixture YAMLs and was not considered green.
