Oktopus 0033 -- fast 2048 startup, separate-piece disk-backed grids
==================================================================

Base: Oktopus 0032 (with earlier 0031 broad-filter patch).
Scope: Oktopus only. No change to neural SDF, DCSDD C++, UDF files,
localization, filtering thresholds, optimization, or YAML.

Changes:
  * The obsolete "valid wrap interval" diagnostic is no longer calculated
    or printed by src/ngc/pwla_curve/_core.py. Defaults valid_s0=0,
    valid_s1=1 remain for API compatibility. Live source has no readers.
  * With --sdf-grid-memmap-dir and separate adaptation pieces
    (combine_adaptations=false): test_dc.py creates a GEOMETRY-ONLY main
    MCGrid. The initial (2049)^3 SDF+mask arrays are never allocated.
  * Per-piece create_grid_like() preserves the runtime MCGrid class and
    scratch directory, rather than falling back to the old RAM-only class.
  * Each per-piece memmap defers initialization until clear_grid(), so
    it is initialized once instead of twice.
  * If combine_adaptations=true: initial memmap fill is deferred to the
    existing clear_grid() in action_part_adapt(); semantics unchanged.
  * If no memmap directory is supplied, initialization behavior is unchanged.

Install from Oktopus root:
    unzip -o ~/Downloads/oktopus-0033-fast-2048-startup-files.zip

No CLI changes. Keep your same --sdf-grid-memmap-dir and --dcsdd-crop-grid.
Note: Changes take effect on a NEW Python process, not an already-running job.

Log for independent pieces:
    [MCGrid] geometry-only grid (no accumulator storage)
    [part_adapt] skipping unused full-grid initialization ...
    [MCGrid] disk-backed SDF grid: ... initialization= deferred

2048 memory facts:
    2049^3 grid: 8,602,523,649 samples
    Float64 SDF values: ~68.8 GB decimal
    Boolean mask: ~8.6 GB decimal
    One per-piece disk-backed file pair is still created and initialized.
    Prefer node-local fast scratch with >=~100 GB free per concurrent grid.
    This does not optimize per-filter 8.6 GB marks or SDF localization.

Tests:
    python -m unittest discover -s tests -p 'test_mcgrid_*py' -v

Tests use small grids; full cluster 2048 runtime and full mesh parity
still need measurement. Do not overwrite locally customized files blindly.
