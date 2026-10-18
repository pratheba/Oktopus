Oktopus 0032 — SDF + DCSDD bridge memory optimization (Oct 18, 2026)
===========================================================================

Updated files: test_dc.py; src/app/agent_3dvec_sdf_dc.py;
src/utils/mc_utils.py. Independent tests in tests/.

This ZIP is based on Oktopus(20261018-041959).zip. Its test_dc.py includes
--dcsdd-fast-bbox-marking from earlier patch 0031. Keep the *other* patch 0031
files (src/app/agent_3dvec_sdf.py, src/ngc/curve_handle.py,
src/ngc/handle_utils/mesh_utils.py) in your local installation.

Default behavior (no new switches):
- Native MCGrid x-fast transfer, no full-grid coordinate/index sorting.
- Exact band-clamped SDF values using an EDT around negative samples.
- SDF model evaluation, localization and geometry parameters unchanged.

Optional switches:
  --sdf-grid-memmap-dir "$TMPDIR"
    Keep the full-resolution float64 SDF grid and marks in scratch-backed
    memmap files. A 1025^3 grid needs ~9.7 GB free scratch. May be slower if
    storage is slow; prefer local fast scratch, not network/Lustre if possible.

  --dcsdd-crop-grid
    EXPERIMENTAL: pass only a padded cubic ROI around all negative samples
    into DCSDD. Preserves zero-crossing support and grid spacing, but can
    change DCSDD's global sphere sampling and therefore final Ours vertices.
    Compare cropped vs uncropped meshes at 128/256 before trusting 1024.

  --dcsdd-full-grid-diagnostics
    Expensive diagnostic percentiles and full-array edge scans are skipped
    for >512 unless this switch is provided. Geometry is unaffected.

These changes do NOT accelerate the SDF network or its curve localization;
they reduce memory in the grid storage and bridge/preprocessing stages.

Install: unzip -o oktopus-0032-sdf-dcsdd-memory-files.zip in Oktopus root.
Run test: python -m unittest discover -s tests -p 'test_*memory.py'
         python -m unittest discover -s tests -p 'test_mcgrid_memmap.py'

The accompanying DCSDD ZIP has a separate README, including rebuild steps.
