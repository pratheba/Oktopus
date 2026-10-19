Oktopus 0042 — enable the 0041 after-cylinder-gate SDF localizer for DIRECT
wrap_radius and rigid_radius adaptations, including combined MC/DCSDD output.

Apply on top of 0041 (or unpack replacement ZIP from repository root).
No changes to YAML, wrap radius formulas, auto snug, tilt, post-SDF seam blending,
individual_support_aware_mc, model batching, or mesh extraction.

Changed:
  test_dc.py
  src/app/agent_3dvec_sdf.py
New:
  tests/test_sdf_wrap_after_gate_0042.py

Example: keep your existing puffer or cone command and add:
  --sdf-localize-after-gate --dcsdd-fast-bbox-marking

Do NOT add --dcsdd-stream-inference / --dcsdd-stream-one-pass /
--dcsdd-stream-crop-first / --dcsdd-stream-replay-metadata to the
supplied puffer/cone YAMLs. Those still use combine_adaptations: true,
blend groups and other per-part features that the independent rigid-boot
streaming implementation intentionally does not handle.

Suggested first test: run 256 or 512, first without and then with these
flags, saving in SEPARATE output directories. Compare per-part accepted
counts, final SDF and mesh. A 1024 run of a combined model may require
considerable dense-grid memory; 0042 does not change that limitation.

New diagnostics:
  [sdf_localize_0042] name= ... mode= direct wrap_radius= True/False ...
  [filter_grid_adapt] ... localize_s= ...

Tests (run in repo):
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_sdf_wrap_after_gate_0042.py' -v
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_sdf_localize_after_gate_0041.py' -v

Limit: 0042 shares ONLY the localization-after-gate optimization with the
combined wrap/rigid path. The other 0035-0040 compact/replay/crop-first
optimizations need a distinct combined-SDF streaming design. No throughput
claim for wrap until it is benchmarked on the real puffer/cone input.
