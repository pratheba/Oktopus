0054 / measured localization-first experiment (apply after 0053)
=================================================================
Goal: accelerate and profile CPU curve projection without changing accepted voxels,
SDF inference, support clamp, avatar cutting, blend-group composition or DCSDD.

Changes:
  * The interval-projection polyline/KDTree may be reused across bounded
    localization batches. The one-entry cache key contains exact byte copies
    of the source curve key parameters and positions. This also catches
    in-place deformations of the skeleton.
  * SciPy KDTree.query can use 2..16 workers for independent spatial queries;
    the default workers=1 retains its original call, including for SciPy
    versions without the workers keyword.
  * Optional per-adaptation logs show projection_s and projection_calls to
    establish how much of localize_compute_s projection actually consumes.
  * No approximate KDTree queries, rounding, quantization, or model changes.

Install on top of your tested 0053 code:
  unzip -o oktopus-0054-localization-projection-files.zip
  or git apply oktopus-0054-localization-projection.patch (NOT both).

Initial geometry audit at 512 (use your existing command unchanged except -y/-o):
  -y adapt_oktopus_puffer_root_leaves_0054_body_audit.yaml \
  -o test_adapt_oktopus_puffer_0054_body_audit_r512
Look for '[sdf_candidate_interval_0048 audit] ... missing_accepted=0'.
Expect accepted=15466060, crop_evaluated=5021570, negative=1381111,
sign_edges=191174, active_cells=189928 and signed_volume~0.143263675.
Strict mesh equality is not guaranteed by counts alone.

After passing audit, profile speed WITHOUT the audit using:
  -y adapt_oktopus_puffer_root_leaves_0054_body_profile.yaml \
  -o test_adapt_oktopus_puffer_0054_body_profile_r512

The supplied t1 YAML executes body plus first three t1 sections, with audits.
Inspect [sdf_projection_0054] projection_s/calls vs total localize_s.
Performance MUST be measured: multiworker KDTree can underperform when
CPUs are oversubscribed or the cost is mainly elsewhere.

New YAML keys:
  sdf_projection_workers_0054: 2  # 1 keeps old query, 2-16 requires SciPy workers support
  sdf_projection_cache_0054: true
  sdf_projection_profile_0054: true
Set all three off (1/false/false) for 0053 behavior.

Tests:
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p test_projection_interval_0054.py -v

Limits:
  - No full 512 neural puffer run completed here.
  - This is a projection cost experiment, not full 2048 streaming,
    full shared t1 localization, or welded root-leaf assembly.
  - These options act on interval projection (default for puffer). Other
    projection modes retain original workers/cache behavior.
  - Use at most the CPUs allocated by your scheduler. Do not use 16 workers
    on a 2-core allocation.
