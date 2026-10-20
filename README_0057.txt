Oktopus 0057 — Audited piecewise source-interval candidate gate
==============================================================
Base: 0056 (the files/patch supplied in the previous turn). Do NOT apply to
0055 without installing 0056 first.

WHY
The 0056 puffer-body at 512: 108,190,890 candidate points, but only
15,466,060 accepted. Original full broadphase was 120,794,571 candidates;
0048a's single interval envelope pruned only 10.43%. Approximate candidate
rejection before the KD-tree is a possible high-impact optimization.

METHOD
0057 divides src_0..src_1 into 12 (configurable 2..32) pieces and builds
an independent tangent-aware localizer gate AABB for each. The candidate
set is the intersection of (a) the ORIGINAL full-cylinder bounding-box
union and (b) the UNION of the piecewise gate AABBs. The implementation
retains ascending unique flat-grid indices, and does not change model
weights, geometry SDF formulas, curve projection, support calibration,
mesh extraction or accepted-row prediction logic.

IMPORTANT: The 0048 and 0057 world envelopes use numerical sampling with
safety guards. They are NOT a mathematical guarantee of the continuous
spline bounds. 0057 requires the exact excluded-voxel localizer audit on
first use, and fails if it excludes even one accepted sample. A subsequent
unaudited profile requires an explicit verified flag. Only reuse that flag
for the EXACT SAME adapted geometry/config/resolution that passed audit.

FILES
src/ngc/handle_utils/bounded_filter.py
src/ngc/handle_utils/interval_envelope.py
src/ngc/curve_handle.py
src/app/agent_3dvec_sdf.py
exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0057_body_audit.yaml
exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0057_body_profile.yaml
tests/test_candidate_piecewise_0057.py
README_0057.txt

INSTALL
cd /fast/pselvaraju/Oktopus_now
git apply --check /path/to/oktopus-0057-piecewise-candidates.patch
git apply /path/to/oktopus-0057-piecewise-candidates.patch
# Or unzip the replacement files from repository root.

FIRST RUN: existing test_dc.py command, with YAML/output changed to:
  -y adapt_oktopus_puffer_root_leaves_0057_body_audit.yaml \
  -o test_adapt_oktopus_puffer_0057_body_audit_r512

DO NOT RUN PROFILE UNTIL AUDIT IS VERIFIED.
Look for:
  [sdf_candidate_piecewise_0057] enabled=True bins=12 ... audit=True
  [sdf_candidate_piecewise_0057 audit] full=... candidates=... \
       pruned=... missing_accepted=0 candidate_reduction_pct=...
  [sdf_localize_0041] ... accepted=15466060 ...
  [dcsdd:grid] sign_changing_edges=191174
  [contouring] Active cells: 189928 / ...
  [dcsdd:ours] ... signed_volume=0.143263675

If missing_accepted > 0, the audit raises UNSAFE: discard the 0057
experiment. If candidates are not materially less than 0056's 108,190,890,
this approach is probably not worth using on puffer.

SECOND RUN: only after successful audit and matching mesh/SDF geometry,
use -y adapt_oktopus_puffer_root_leaves_0057_body_profile.yaml
 -o test_adapt_oktopus_puffer_0057_body_profile_r512
The profile has sdf_candidate_interval_audit: false and explicit
sdf_candidate_piecewise_verified_0057: true. This flag is NOT proof by
itself: it is your declaration of prior audit for unchanged configuration.
Compare profile total time with 0056's 204.587s, localize_compute_s=115.821,
projection_s=55.763, and filter_s=139.147. Do not compare audited time
alone because the audit re-localizes excluded candidates.

ROLLBACK
Set sdf_candidate_piecewise_0057: false (or use existing 0056 YAML).
Defaults are inert. The original 0048a single envelope remains available.

TESTS
PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_candidate_piecewise_0057.py' -v
PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_sdf_candidate_interval_0048.py' -v
PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_puffer_staged_candidate_roi_0052.py' -v

STATUS
Synthetic code/test parity checked. No real puffer geometry, HPC speed-up
or 2048 memory/run-time improvement has been validated for 0057 yet.
