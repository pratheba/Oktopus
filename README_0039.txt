Oktopus 0039 — Skip the redundant accepted-voxel counting pass when the
accepted count is already known from a completed run with the SAME configuration.

BASED ON: 0038 (0031–0038 installed). Only 3 source/test files changed/added:
  test_dc.py                                  updated
  src/app/agent_3dvec_sdf.py                 updated
  tests/test_dcsdd_stream_skip_count_2048.py NEW

INSTALL from the Oktopus root:
  unzip -o ~/Downloads/oktopus-0039-exact-count-skip-files.zip

TEST:
  PYTHONPATH=src:src/ngc:tests python -m unittest \
    tests.test_dcsdd_stream_skip_count_2048 \
    tests.test_dcsdd_stream_replay_2048 \
    tests.test_dcsdd_stream_crop_2048 -v

RUN: Use the same successful 0038 command but add EXACTLY ONE of:
  --dcsdd-stream-accepted-count 376525454    # Boot 1, resolution 2048
  --dcsdd-stream-accepted-count 47383570     # Boot 1, resolution 1024
The two values above are measured from the user's validated earlier runs.
Do not reuse one with a different resolution or unrelated adaptation.

REQUIRES: an independent, single rigid Boot 1 adaptation, --dcsdd-bounded-filtering,
--dcsdd-stream-inference, --dcsdd-stream-crop-first,
--dcsdd-stream-replay-metadata, --dcsdd-crop-grid, and the other 0038 flags.

WHY AN EXPLICIT COUNT?
  The original inference uses np.array_split(N, N//batch_size+1), so exact
  inference chunk boundaries require KNOWING N beforehand. Fully dynamic
  inference without counting changes these boundaries; staging all input
  coordinates until the count is known would consume substantial scratch.
  0039 reuses an exact previously measured count without doing the first
  expensive geometry localization pass. The sole prediction pass STILL
  counts accepted indices and checks equality BEFORE support calibration,
  cropping or DCSDD meshing. A mismatch stops execution with an error.

NO FLAG = 0038 unchanged, including safe counting pass.

EXPECTED:
  [part_adapt stream] accepted-count hint= 376525454 ...
  [part_adapt stream] ... count_s= 0.000 ...
  [part_adapt stream] accepted-count verified= 376525454 ... saved_count_pass=True

This is a targeted high-resolution Boot 1 optimization, not a general
count-free streaming implementation for new shapes or configurations.
It does not touch DCSDD C++, UDF, wrap radius, YAML geometry or SDF equations.
Expected time saving is the previously measured count_s (~875s at 2048),
subject to real cluster load; end-to-end performance remains to be measured.
