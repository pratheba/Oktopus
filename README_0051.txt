0051 — Minimal, correction-preserving bounded puffer localization staging
=======================================================================
Prerequisite: 0048a -> 0049 -> 0050 -> 0050a. Apply this patch after 0050a.

The 512 puffer-body log reported:
  candidates=120794571 accepted=15466060
  localize_compute_s=208.292 stage_write_s=309.301 staged_GB=5.815

Cause: the 0050a bounded staging writes all returned curve fields from both
avatar and accessory (including frame, points, world-space samples, etc.) and
stores `inside` indices that the puffer action never consumes.

0051 adds opt-in YAML setting:
  sdf_puffer_minimal_staging: true

For each accepted sample it preserves (with the ORIGINAL dtype and values):
  accessory: samples_local, coords, angles, radius, rho, rho_n
  avatar when avatar SDF is needed: the same six keys
  avatar otherwise: coords (required by query/post-SDF seam blending)
  grid kidx: unchanged int64 IDs in unchanged order
The candidate-relative `inside` array is deliberately not saved. The SDF
neural models' input dictionaries, wrap/rigid mapping, avatar cut, snug, seam
warps, blend composition, mesh extraction and UDF ownership remain unchanged.
For model inference, contiguous memmap slices avoid an unnecessary np.arange
index allocation while retaining the exact np.array_split batch boundaries.

Expected puffer-body stage PAYLOAD at 512: about 2.35 GB versus measured
0050a staged 5.815 GB (six float64 fields per curve + int64 kidx).
That is a calculated size, not a measured write time or peak RAM result.
The code still uses disk-backed bounded staging and still visits 120.8M
candidates; this is NOT the final 2048-ready streaming evaluator.

INSTALL:
  cd /fast/pselvaraju/Oktopus_now
  unzip -o ~/Downloads/oktopus-0051-puffer-minimal-stage-files.zip
  # Or use the .patch alone; do not apply ZIP AND patch.

RUN:
  Use the SAME puffer 512 command (and -y adapt_oktopus_puffer_root_leaves_0050.yaml)
  as your 0050a run, with bounded filtering enabled. No new CLI flag.
  Keep --dcsdd-stream-inference, --sdf-stream-unified, and unaudited
  interval pruning OFF for puffer. Use 512; do not attempt 2048 yet.

EXPECT:
  [sdf_puffer_0051] name= puffer_body_wrap_no_snug minimal_correction_staging=True
  [filter_grid_adapt] ... after_localize=15466060 ... stage_write_s=... staged_GB=...
  [sdf_puffer_0051] minimal_staging=True avatar_model_inputs=True ...
  Geometry invariants: accepted body count 15466060, matching SDF sample
  and DCSDD statistics from 0050a before attempting 1024.

DISABLE / ROLLBACK:
  Set sdf_puffer_minimal_staging: false (or remove the line) to restore 0050a
  staging, including full localization fields and the old inside array.

TEST:
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
     -p test_puffer_minimal_staging_0051.py -v

VALIDATION:
  Unit parity is exact for all six model inputs, avatar-only source coords,
  accepted voxel indices and dtypes on synthetic wrap/rigid cases including
  avatar-cut and detail-avatar-gate cases. Previous puffer, 0042, 0048, 0047
  and root-leaf targeted tests passed. The real neural 512 performance and
  full-scene mesh still need your live run. 2048 remains unverified.
