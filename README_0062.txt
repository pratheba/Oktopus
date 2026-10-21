Oktopus 0062 — conservative pre-query KDTree broad phase
========================================================

Purpose
-------
0061 512 body profile: 144.0287 s total / 56.883 s localization,
including 37.937 s KD-tree lookup. The existing 0058 distance gate
rejects 37,572,350 of 106,630,916 candidates, but 0058 tests that
distance only AFTER exact KD-tree nearest-neighbor lookup.

0062 adds an opt-in coarse Cartesian occupancy grid around the cached
interval skeleton. It can discard some candidates BEFORE the costly
nearest-neighbor query. For each coarse cell it measures the exact
nearest-skeleton-vertex distance at the cell center. A cell is excluded
only if center_distance > existing_0058_bound + cell_half_diagonal + epsilon.
The triangle inequality guarantees every excluded point also fails the
existing 0058 nearest-vertex distance bound. Nothing about the cylinder,
wrap-radius, SDF formulas, support calibration, inference or mesher changes.
The survivor row mapping is preserved in original order.

An optional, independent exact SciPy distance_upper_bound query can
reduce KD-tree traversal even for remaining points. It is OFF in the
initial audit/profile configurations because its benefit is data-dependent.
When enabled, nextafter(bound, +inf) protects 0058's <= boundary rule.

The grid is built once per interval tree/bound and reused across batches.
It is capped at 250,000 cells; if unsupported or too large it falls back
to the unchanged full query (never an unsafe approximation). Feature is
restricted to the already-audited 0058 direct k=1 interval path.

Files
-----
src/ngc/pwla_curve/_geometry.py
src/ngc/pwla_curve/_localize.py
src/app/agent_3dvec_sdf.py
exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0062_body_audit.yaml
exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0062_body_profile.yaml
exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0062_body_bounded_audit.yaml
exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0062_body_bounded_profile.yaml
tests/test_projection_grid_0062.py
README_0062.txt

Apply ON TOP OF your working 0061 tree
--------------------------------------
cd /fast/pselvaraju/Oktopus_now
git apply --check ~/Downloads/oktopus-0062-prequery-grid.patch
git apply ~/Downloads/oktopus-0062-prequery-grid.patch

Keep the same test_dc.py command and change only -y and -o.
FIRST: -y adapt_oktopus_puffer_root_leaves_0062_body_audit.yaml
       -o test_adapt_oktopus_puffer_0062_body_audit_r512
This audits EVERY projected candidate batch against the original exact
full-KDTree query, comparing kept ORIGINAL indices and curve parameters
with numpy array_equal (not an approximate tolerance). The full query
runs as a second reference, so the audit run is not a speed benchmark.

IF PASS: -y adapt_oktopus_puffer_root_leaves_0062_body_profile.yaml
         -o test_adapt_oktopus_puffer_0062_body_profile_r512

If the coarse grid alone helps, optionally test the bounded query:
  -y adapt_oktopus_puffer_root_leaves_0062_body_bounded_audit.yaml
  -o test_adapt_oktopus_puffer_0062_body_bounded_audit_r512
and ONLY if that passes:
  -y adapt_oktopus_puffer_root_leaves_0062_body_bounded_profile.yaml
  -o test_adapt_oktopus_puffer_0062_body_bounded_profile_r512

Invariants for this particular 512 body:
  candidates=106630916; accepted=15466060; sphere early rejection=37572350
  sign_changing_edges=191174; active_cells=189928; volume~0.143263675
  original geometry remains non-watertight (unrelated to 0062).

0062 log (per adaptation):
  [sdf_projection_grid_0062] candidates=... knn_queried=...
  prequery_skipped=... grid_cells=... grid_build_s=...
  grid_filter_s=... grid_fallbacks=... bounded_query=...
  audited_rows=...
The old [sdf_projection_0055] knn_s timer measures *queried* points;
the new grid_build_s and grid_filter_s expose the additional overhead.
Compare TOTAL time against 0061 non-audit baseline 144.0287 s; fewer
KD-tree rows alone do not prove a faster overall run.

Local validation
----------------
0062 tests pass exact projection / localization parity, full wrap and
rigid adaptation field parity, radius variants, empty/all-far inputs,
cache invalidation, fallback, YAML forwarding and 0062 audit correctness.
Earlier targeted 0054–0059 regression tests pass in the reconstructed
source except two pre-existing 0055 configuration tests whose older YAML
files are absent from the reconstructed 0061 workspace, and an older
0041 fixture that already does not accept 0054 projection arguments.
No real 512 puffer performance result is claimed yet.
