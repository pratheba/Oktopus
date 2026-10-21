Oktopus 0060 — opt-in localization diagnostic timing (NO algorithm change)
=====================================================================

WHAT IT MEASURES
----------------
This is a diagnostic-only follow-up to 0059 (the fastest body profile in
this sequence: 180.0367 s at resolution 512, with audit disabled).

The existing log numbers:
    [sdf_localize_0041] localize_s=98.136
    [sdf_localize_reuse_0059] core_s=49.243 avatar_s=1.355
    [sdf_projection_0054] projection_s=42.920
must not be summed: they are nested and cover different scopes. The new
0060 profiler reports EXCLUSIVE time within each scope, so we can identify
what happens outside initial curve projection.

New log lines are emitted ONLY with sdf_localize_profile_0060: true:
  [sdf_profile_0060] scope=stream ...
  [sdf_profile_0060] scope=adapt exclusive_sum_s=... stages=...
  [sdf_profile_0060_stage] scope=adapt stage=... seconds=...
  [sdf_profile_0060] scope=core exclusive_sum_s=... stages=...
  [sdf_profile_0060_stage] scope=core stage=... seconds=...

STREAM scope (wall, counts, batches, p50/p95 batch wall time):
  candidate_iterator_s   generator advancement, candidate gate/index creation
  decode_s               MCGrid voxel-index-to-position conversion
  localize_s             localize_samples_adapt calls (includes core & adapt)
  audit_s                parity audits, if enabled
  selection_s            accepted local -> grid index gather
  callback_s             0053 inference/writer callback

CORE scope (inside localize_samples):
  projection                interval curve projection (contains KD-tree/refine)
  projection_select         projection range and candidate selection
  frame_interpolation       avatar 3D points and frame matrices
  world_to_local            world -> frame einsum
  cylinder_radius_interpolation  cylinder + tangent radius fields
  cylinder_gate             norm/gate and accepted-row selection
  train_radius_interpolation  accepted train radii
  output_gather             gather selected fields into output dictionary

ADAPT scope (inside localize_samples_adapt):
  setup                        update target coordinates/frame and scalar setup
  avatar_localize              call full CORE (NESTED; do not add to CORE)
  source_interval_crop         restrict avatar source interval, slice fields
  avatar_fields                selected physical and normalized avatar coords
  arclength_mapping            map source to target arclength
  optional_ring_exports        optional debug ring work (normally ~0)
  source_uv_center             source center adjustment
  wrap_table_preparation       prepare wrap radius tables and other setup
  accessory_interpolation     interpolate target frame/position/radii/center
  radial_mapping              angular wrap/rigid/other radius mapping
  physical_local_coordinates  accessory physical local coordinate construction
  normalize_model_coordinates accessory model input vectors and trig/tilt
  pack_output_fields          output accessory dictionary

Within a scope, these are exclusive non-overlapping stages; scope totals may
have small timing gaps from instrumentation and are NOT additive across scopes.
The existing [sdf_puffer_0053] reports model_s/prediction_write_s, and
[dcsdd] and [part_adapt timing] report mesh and outer stages. These remain
unchanged. Sampling profiler overhead may slightly increase wall-clock time.

INSTALL (requires 0059 already applied)
---------------------------------------
    cd /fast/pselvaraju/Oktopus_now
    git apply --check ~/Downloads/oktopus-0060-localization-profiling.patch
    git apply ~/Downloads/oktopus-0060-localization-profiling.patch

Or unzip oktopus-0060-localization-profiling-files.zip over project root.

RUN BODY PROFILE AT 512
-----------------------
Keep your existing python test_dc.py command, replacing just:
    -y adapt_oktopus_puffer_root_leaves_0060_body_profile.yaml \
    -o test_adapt_oktopus_puffer_0060_body_profile_r512

Both 0058 and 0059 parity audits remain OFF (both passed previously for this
configuration). 0057 candidate pruning, 0058 early rejection and 0059 reuse
stay ON. This stops after part 1 of 15; there is no final root/leaf export.

Expected unchanged: 106,630,916 candidates; 37,572,350 sphere early rejects;
15,466,060 accepted; 191,174 sign-changing edges; 189,928 active cells;
signed volume ~0.143263675. Geometry stats alone are not a vertex-level audit.

Collect the new [sdf_profile_0060] lines together with existing [sdf_projection],
[sdf_localize_0041], [sdf_puffer_0053], [part_adapt timing], and [dcsdd].
We'll choose the next optimization from the dominant EXCLUSIVE adapt stage,
not from guesses based on nested timers.

For runtime comparisons, turn sdf_localize_profile_0060 OFF with the same
configuration and compare to the non-profile 0059 baseline (180.04 seconds).

VALIDATION / ROLLBACK
---------------------
0060 is opt-in. Disable sdf_localize_profile_0060 to deactivate all new
profiling without changing any optimization setting. Local parity testing
passes for rigid and wrap, empty points and bounded callback stream. The 0059
regression and 0053 prediction-stream / 0054 projection tests also pass.
The real 512 puffer run has not yet been executed here.
