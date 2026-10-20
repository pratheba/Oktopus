Oktopus 0055 — exact cached adjacent-edge refinement + root/leaf-only output
=========================================================================
Base: 0054a, including all earlier 0049–0053 patches. Apply 0055 exactly once.

SCOPE
  - sdf_projection_refine_fast_0055: true performs EXACTLY the prior per-
    nearest-vertex adjacent-edge XOR projection, but gathers precomputed
    segment length/direction/timing instead of rebuilding those arrays for
    each of hundreds of localization batches.
  - sdf_projection_profile_split_0055: true reports nearest-neighbor query
    wall time (`knn_s`) separately from adjacent-edge refinement (`refine_s`).
  - Existing 0054 KD tree cache and workers remain available; no new packages.
  - 0055 is opt-in and leaves other SDF/UDF paths unchanged by default.
  - Full-audit and t1-audit YAMLs set export_individual_meshes: false to
    skip redundant section-by-section DCSDD extraction. Root/leaf singleton
    parts and completed blended group meshes are STILL extracted by the
    existing 0050 logic and are the UDF transfer inputs (`riggable_parts`).
    If you need intermediate section meshes for visual diagnosis, set it true.
  - No welded root-leaf surface or animation added.

INSTALL
  cd /fast/pselvaraju/Oktopus_now
  unzip -o ~/Downloads/oktopus-0055-exact-fast-projection-files.zip

FIRST RUN (512, body-only, audited)
  Use all your working test_dc.py flags, replacing only:
   -y adapt_oktopus_puffer_root_leaves_0055_body_audit.yaml
   -o test_adapt_oktopus_puffer_0055_body_audit_r512
  Do not run two jobs writing the same output folder or scratch directory.

  Look for:
   [sdf_candidate_interval_0048 audit] missing_accepted=0
   [sdf_projection_0055] fast_refine=True knn_s=... refine_s=...
   [sdf_projection_0054] ... projection_s=...
  Reference (0054a at 512):
   candidates=108190890 accepted=15466060; evaluated=5021570;
   negative=1381111; sign_edges=191174; active_cells=189928;
   signed_volume=0.143263675; total=221.79 sec.
  Compare values/indices and final mesh before disabling the audit.

SECOND RUN (body-only time without audit, after parity)
   -y adapt_oktopus_puffer_root_leaves_0055_body_profile.yaml

T1 CHECKPOINT (body + 3 t1 sections, audited, no intermediate meshes)
   -y adapt_oktopus_puffer_root_leaves_0055_t1_audit.yaml
  t1 final group extraction still occurs. Compare against the previous
  t1 output (evaluated=3625202, sign edges=40910, active_cells=40830).

FULL 512 (only after smaller audits pass)
   -y adapt_oktopus_puffer_root_leaves_0055_full_audit.yaml
  This checks new sections with fail-closed pruning. If the audit fails for a
  section, do NOT disable the audit to force it through. No 2048 run yet.
  The output root/leaf static assembly remains UNSTITCHED by design.

LOCAL TESTS
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
    -p 'test_projection_refinement_0055.py' -v
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
    -p 'test_projection_config_0055.py' -v

PERFORMANCE EVIDENCE
  A local synthetic 120k-point projection benchmark (not puffer, different
  hardware) found median adjacent-edge refinement 20.7ms -> 13.4ms,
  with both producing identical float64 projection outputs. Full interval
  projection median 60.8ms -> 53.6ms. Actual puffer improvement is unmeasured;
  KD-tree query and other localization still dominate, so avoid extrapolating
  these microbenchmarks to end-to-end speed.

ROLLBACK
  Set sdf_projection_refine_fast_0055: false, or use original 0054a YAML.
  Set export_individual_meshes: true if per-section diagnostic PLYs are needed.
