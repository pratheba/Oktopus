Oktopus 0034 -- memory-bounded 2048 boots filtering/localization
================================================================
Base: Oktopus 0033 with 0031 and 0032 already installed. No DCSDD C++
changes and no UDF file changes. The new flag is OPT-IN.

WHAT CHANGES
- Replaces the full (reso+1)^3 broad-phase Boolean mask with bounded
  Z-slab marking of the EXACT same union of cylinder bounding boxes.
- Generates candidates in ascending x-fast grid order in fixed batches.
- Runs direct rigid-radius localization in candidate batches and retains
  the same selected voxel indices and per-point data order.
- Uses scratch-backed arrays to hold accepted localization data for
  large grids when --sdf-grid-memmap-dir is specified.
- Existing unbounded behavior is unchanged without the new flag.

INSTALL FROM OKTOPUS ROOT
  unzip -o ~/Downloads/oktopus-0034-bounded-2048-files.zip

TESTS
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_dcsdd_bounded_filtering.py' -v

CLI: add this to the previously working 2048 boots command:
  --dcsdd-bounded-filtering --dcsdd-filter-batch-size 262144
Keep --dcsdd-fast-bbox-marking, --dcsdd-crop-grid, and
--sdf-grid-memmap-dir. A writable scratch directory with ample space is
required; this implementation stages per-sample localization data on disk
as well as each individual full SDF grid.

SUPPORTED NOW: direct rigid_radius boots. Wrap/adaptive modes that compute
batchwide statistics are not enabled because chunking could affect geometry.

VERIFICATION: tests cover candidate index parity and data order on small
grids, with and without scratch staging. They do NOT establish that
2048 runs within a 100 GiB cgroup: test in HTCondor, monitor memory and
scratch, and retain the 1024 reference meshes for geometry comparison.
