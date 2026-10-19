Oktopus 0040 — Automatic one-pass accepted-count streaming
===========================================================

Requirements: 0038 + 0039 installed. This ZIP contains only NEW/UPDATED files.

Copy files into your Oktopus repo at their existing paths:
  unzip -o ~/Downloads/oktopus-0040-one-pass-files.zip

New flag: --dcsdd-stream-one-pass
Required existing flags:
  --dcsdd-bounded-filtering
  --dcsdd-stream-inference
  --dcsdd-stream-crop-first
  --dcsdd-stream-replay-metadata
  --dcsdd-crop-grid
  --sdf-grid-memmap-dir /path/to/writable/scratch

Do NOT supply --dcsdd-stream-accepted-count with one-pass mode.
The new flag is opt-in and currently valid only for a SINGLE independent
rigid-radius Boot 1-style adaptation with global accessory support clamp.
Puffer wrap-radius, Boot 2 tiled detail, and multiple adaptations in a single
YAML invocation remain unsupported in this mode.

What changes:
 * No accepted-voxel pre-count, no manual accepted-count hint.
 * One bounded candidate localization pass.
 * Fixed accepted-sample inference batches (default 262144), final partial.
 * Appends prediction scalars and minimal replay metadata in ordinary
   buffered binary files; no preallocated N-dependent memmaps.
 * Global support calibration uses all predictions, then identical crop-first
   replay math, output indexing and DCSDD C++.

Numerical caution: old inference used np.array_split based on FINAL accepted
count; the new fixed batch boundaries differ. Mathematically pointwise inference
is unchanged. GPU numerical output can differ slightly by kernel/batch shape.
Compare 1024 results with the 0038/0039 baseline before using 2048 output.

Test:
 PYTHONPATH=src:src/ngc:tests python -m unittest \
   tests.test_dcsdd_stream_one_pass_2048 \
   tests.test_dcsdd_stream_skip_count_2048 \
   tests.test_dcsdd_stream_replay_2048 \
   tests.test_dcsdd_stream_crop_2048 \
   tests.test_dcsdd_stream_2048 \
   tests.test_dcsdd_bounded_filtering -q

Expected diagnostics:
 [part_adapt stream] count_source=one_pass_auto
 [part_adapt stream] candidates=... accepted=... count_s=0.000 partition=fixed_accepted_batch
 [part_adapt stream timing] ... total_s=...

For each new shape/resolution, no accepted count has to be supplied.
If scratch storage fills, normal file writes raise OSError rather than SIGBUS
from an incorrectly-sized writable memmap. Sufficient disk space is still needed.
