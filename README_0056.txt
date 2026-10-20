Oktopus 0056 — KD-tree leafsize trial + optional fast prediction spool
======================================================================
Base: the exact uploaded 0055 files from Oktopus(20261020-133616).zip.

WHY
0055 body audit (512): knn_s=60.212, refine_s=7.786, total=242.199.
0054a body audit (512): projection_s=70.920, total=221.788.
Even though 0055 improved projection to 68.161 s, the write_s grew
from 1.388 to 23.577 s. This is consistent with I/O contention; it is
not sufficient evidence of a 0055 write-path regression.

CHANGES
1) sdf_projection_leafsize_0056: 32 (opt-in; default remains 10).
   This changes only SciPy KDTree leaf size, not eps or mathematical query,
   but exact equidistant nearest-vertex ties can pick a different index.
   Tree cache key includes leafsize. For full geometry equivalence validate
   candidate/accepted counts and extracted mesh before committing.
2) sdf_prediction_zero_copy_write_0056: true (opt-in; false by default).
   Writes the same C-contiguous bytes through memoryview without allocating
   full temporary bytes objects; reference and new bytes match in tests.
3) sdf_prediction_scratch_dir_0056: optional existing writable path.
   Overrides the temp spool only; it does not relocate MCGrid / network
   weights / outputs. Useful if network-backed memmap scratch is slow. The
   temporary spool is closed/removed after adaptation processing.
   DO NOT enable without verifying that your HTCondor compute node has a
   node-local disk with enough space: body at 512 writes about 1.1 GB;
   other sections/resolutions may need much more.

INSTALL (from Oktopus project root, against exact uploaded 0055)
  git apply --check /path/to/oktopus-0056-io-kdtree.patch
  git apply /path/to/oktopus-0056-io-kdtree.patch
OR extract oktopus-0056-io-kdtree-files.zip at the project root.

RUN (first: original geometry audit at 512, body only)
  Keep all your previously working test_dc.py options, change just:
    -y adapt_oktopus_puffer_root_leaves_0056_body_audit.yaml \
    -o test_adapt_oktopus_puffer_0056_body_audit_r512
  Confirm:
    [sdf_candidate_interval_0048 audit] ... missing_accepted=0
    [sdf_projection_0054] ... leafsize_0056=32 ...
    [sdf_projection_0055] ... knn_s=... refine_s=...
    [sdf_puffer_0053] ... zero_copy_0056=True ...
  Compare to 0055: accepted=15466060, sign_edges=191174,
    active_cells=189928, signed_volume=0.143263675.
  For a fair speed comparison without candidate audit use 0056_body_profile.

OPTIONAL NODE-LOCAL SPOOL (separate test; NOT enabled in supplied YAMLs)
  Inside your HTCondor job, check:
    echo "TMPDIR=${TMPDIR:-} _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-}"
    df -h "${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
  If suitable, add to the 0056 YAML (with environment present in the job):
    sdf_prediction_scratch_dir_0056: "$_CONDOR_SCRATCH_DIR"
  Or provide a verified absolute local scratch path. If the variable is
  missing or the directory does not exist, the run fails early.

LOCAL TESTS
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p '*0056.py' -v
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_projection_refinement_0055.py' -v
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_projection_forwarding_0054a.py' -v

ROLLBACK
  sdf_projection_leafsize_0056: 10
  sdf_prediction_zero_copy_write_0056: false
  Remove sdf_prediction_scratch_dir_0056 or leave it unset.

CAUTION
Synthetic local KDTree benchmarks favored leafsize=32, but the actual 512
puffer improvement has NOT been measured. Changing leafsize can change
nearest-vertex tie resolution. Do not assume byte-identical output solely
from active-cell/volume matching; compare final SDF/mesh or hold 0055 as
validated baseline. No 2048 run is recommended until 512 parity confirmed.
