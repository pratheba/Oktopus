0038 -- Opt-in replay metadata for independent rigid Boot 1 crop-first DCSDD

BASED ON: Oktopus 0037, with earlier 0031-0036 already installed.
This archive contains ONLY four updated/new files:
  test_dc.py                                      updated
  src/app/agent_3dvec_sdf.py                      updated
  src/app/dcsdd_stream_replay.py                  NEW
  tests/test_dcsdd_stream_replay_2048.py          NEW

From the Oktopus repository root, extract the ZIP, preserving paths:
  unzip -o ~/Downloads/oktopus-0038-replay-metadata-files.zip

Test:
  PYTHONPATH=src:src/ngc:tests python -m unittest \
    tests.test_dcsdd_stream_replay_2048 \
    tests.test_dcsdd_stream_crop_2048 -v

Keep your validated independent Boot 1 YAML. Run the same 0037 1024
command with one additional flag:
  --dcsdd-stream-replay-metadata

OPTIONAL: if the compute node has node-local scratch with enough FREE space:
  --dcsdd-stream-replay-dir "$TMPDIR"
This directory is used only for 3 scalar prediction buffers and the 24-byte
(index, coord, rho) record per accepted sample. The SDF grid remains under
--sdf-grid-memmap-dir. Omitting this flag uses the existing memmap directory.
At 1024 Boot 1 the accepted count is 47,383,570: metadata ~1.137 GB,
predictions ~0.569 GB (float32). The preflight conservatively requires ~2.39 GB.
At 2048 this increases ~8x; check scratch free space and job RAM/cgroup.

Mechanics:
  count pass: unchanged, for exact global model partition sizes.
  inference pass: unchanged neural input/partition and global quantiles; also
    write accepted global index, curve coordinate and exactly-computed rho.
  bbox replay: read predictions + metadata, compute exact support clamp, find
    negative bbox WITHOUT localizing again.
  final write replay: compose same SDF and write only cropped voxel indices
    WITHOUT localizing again.
  DCSDD C++, wrap radius, Boot 2 tiled detail, UDF: UNCHANGED.

Diagnostic lines:
  [part_adapt stream replay] metadata_GB= ... metadata_write_s= ...
  [part_adapt stream crop] bbox replay chunks= ...
  [part_adapt stream crop] write replay chunks= ...
  [part_adapt stream timing] ...

Expected parity: negative bbox, cropped SDF, sample count, active cells, faces.
Exact geometry and speed still require your 1024/2048 cluster validation.
To revert to 0037, omit --dcsdd-stream-replay-metadata; old path is unchanged.
