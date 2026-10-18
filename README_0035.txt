0035 - Compact disk staging for independent rigid boots at 2048
================================================================
BASE: Oktopus 0034 (with 0033/0032/0031 previously installed).

This archive contains complete replacements, preserving directory paths.
Extract from Oktopus root:
  unzip -o ~/Downloads/oktopus-0035-compact-2048-boot1-files.zip

Run the two tests:
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_dcsdd_compact_2048.py' -v
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_dcsdd_bounded_filtering.py' -v

CLI: add --dcsdd-compact-staging to existing 2048 boots command;
retain --dcsdd-bounded-filtering, --dcsdd-fast-bbox-marking,
--dcsdd-crop-grid, --sdf-grid-memmap-dir, and 5/3 iterations.

This is OPT-IN and will activate for Boot 1 (direct, independent,
rigid radius, no avatar cut/snug, no tiled detail, no seam warp).
Boot 2 uses tiled detail, so it AUTOMATICALLY falls back to the 0034
full localization staging path. It is not an error.

It removes unused avatar fields and candidate-relative inside indices
from disk staging for eligible boots, retaining the exact model inputs:
 samples_local, coords, angles, radius, rho, rho_n.
It also avoids a giant full-length np.arange() in inference and
uses identical np.array_split batch boundaries through contiguous slices.

The reported localize_s includes curve localization and scratch writes.
0035 prints localize_compute_s, stage_write_s, and staged_GB to allow
us to measure actual bottlenecks rather than extrapolate from total time.

Only tested on local synthetic geometries / inference stand-ins, NOT on
full 2048 actual model, and may or may not speed up depending on Lustre IO.
No changes to UDF files, YAML, or DCSDD C++.
