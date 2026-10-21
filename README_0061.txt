Oktopus 0061 — vectorized angular + longitudinal wrap-radius interpolation
=======================================================================

GOAL
----
0060 profiling identified two expensive EXCLUSIVE adaptation stages in the
512 puffer body: avatar_localize=48.678 s and radial_mapping=41.772 s.
The hot wrap function interpolate_wrap_radius1 builds an (N, Ns) angular
radius table and calls np.interp once PER POINT in a Python for loop.
Both source and target curves invoke it for each accepted batch.

0061 replaces that path with O(N) batched four-corner interpolation. Angular
periodicity is calculated with the ORIGINAL interpolate_theta arithmetic,
and s is interpolated between adjacent nonuniform s_bins, using the same
endpoint clamps as np.interp. No changes to projection, SDF formulas, wrap
scales, candidate sets, MC/DCSDD, or the other already-vectorized steps.

For bad/non-monotone s bin tables, 0061 falls back to the original routine.
This is opt-in and fully reversible; default legacy behavior is unchanged.

FILES
-----
  src/ngc/curve_functions/_interpolate.py
  src/ngc/pwla_curve/_localize.py
  src/ngc/curve_handle.py
  src/app/agent_3dvec_sdf.py
  exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0061_body_audit.yaml
  exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0061_body_profile.yaml
  tests/test_wrap_vectorized_0061.py
  README_0061.txt

INSTALL on top of 0060 (0060 includes 0056–0059):
  cd /fast/pselvaraju/Oktopus_now
  git apply --check ~/Downloads/oktopus-0061-wrap-vectorization.patch
  git apply ~/Downloads/oktopus-0061-wrap-vectorization.patch

FIRST RUN (AUDIT):
Use the exact same python test_dc.py command as for 0060, replacing just:
  -y adapt_oktopus_puffer_root_leaves_0061_body_audit.yaml \
  -o test_adapt_oktopus_puffer_0061_body_audit_r512

Audit YAML enables:
  sdf_vectorize_wrap_0061: true
  sdf_vectorize_wrap_audit_0061: true
  sdf_localize_profile_0060: true
It compares ALL source+target wrap-radii against the legacy per-point code
for every adaptation batch, with rtol=1e-12 and atol=1e-12, equal_nan=True.
Mismatches raise immediately. There will be substantially more audit runtime;
do NOT use this run to compare speed. Floating-point operations can differ by
last-bit rounding; this does not establish bitwise equality.

SECOND RUN (PROFILE, ONLY IF AUDIT PASSES):
  -y adapt_oktopus_puffer_root_leaves_0061_body_profile.yaml \
  -o test_adapt_oktopus_puffer_0061_body_profile_r512

This disables the 0061 audit AND 0060 instrumentation for a fairer runtime
comparison with the 0059 profile baseline (180.0367 s). 0057–0059 fast paths
remain enabled, as in the previously audited puffer benchmark.

Expected invariant counts for this exact 512 body:
  candidates=106630916; accepted=15466060; early rejected=37572350;
  sign_changing_edges=191174; active_cells=189928;
  signed_volume about 0.143263675
Full mesh equality is NOT proven by those aggregate checks.

NEW FINAL LOG
-------------
  [sdf_wrap_vector_0061] curve=source calls=... points=...
      fallback_calls=... audited_points=... max_abs_error=...
  [sdf_wrap_vector_0061] curve=target ...
The source and target should have 428 calls each and together process about
2x15.47 million accepted points (if no extra adaptations). The stat fields
are cumulative on each core instance, not per-dataset historical global data.

LOCAL VALIDATION
----------------
59 targeted tests passed across 0053–0061, including full synthetic wrap
adaptation numerical parity of every returned field and accepted index;
nonuniform s bins, endpoints, far-outside queries, periodic theta; duplicate
s-bin fallback; and existing streaming/projection regressions.

30,000 points, 200 x 120 wrap table (local synthetic warm timing):
  legacy: ~0.139 s; vectorized: ~0.009 s (about 15x on that isolated step)
  max abs difference: 2.22e-16
No real 512 puffer speedup is claimed until its non-audit profile run.

Note: the original DCSDD mesh on this puffer is not watertight. This patch
is a performance optimization and does not attempt to fix mesh topology.
