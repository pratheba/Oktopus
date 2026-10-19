Oktopus 0049 — puffer sparse blend-group staging (experimental)
==================================================================
Builds on 0048a. Use replacement ZIP *or* git patch, not both.

What is implemented:
- Opt-in sdf_puffer_sparse_groups for existing 15-adaptation puffer root/leaves.
- Preserves current per-adaptation localization, wrap/rigid mapping, inference,
  avatar SDF carving, hard/soft snug, adaptive shell treatment and blend rules.
- Avoids allocating and filling a FULL 3D grid for EVERY individual DCSDD
  extraction and every multi-section tentacle's SDF group. Accumulates group
  SDF on sorted unique global indices, then constructs ONLY the exact padded
  negative-SDF cube and feeds it to DCSDD ours.
- Keeps existing per-section meshes for UDF cut.
- No global leaf/leaf union. This does NOT stitch body/tentacle junctions.
- 0048a interval pruning is intentionally disabled in the supplied puffer YAML
  until a per-adaptation 512 audit establishes that it is safe.

Important limitation:
- Per-adaptation localization/inference still uses the original correction-
  rich combined path, including bounded localization staging. This is not the
  fully unified count-free 0040 streaming evaluator for every puffer section.
  It does NOT guarantee acceptable runtime or memory at 2048 until profiled.
- Sorted sparse arrays are retained in RAM per completed group and their
  merge temporarily allocates more RAM. Measure maximum RSS and scratch use.
- Only tested with synthetic numerical SDF/geometry tests, NOT with live puffer
  checkpoints. Do NOT run 2048 before a 512 parity inspection.

Run initial SDF test (on top of 0049, before 0050):
  existing test_dc.py command, selecting
  -y adapt_oktopus_puffer_root_leaves_0049_stream.yaml
  -r 512 -a sdf_dc --surface-extraction ours --dcsdd-methods ours
  --sdf-grid-memmap-dir <scratch-directory>

Validate per-group overlap counts, individual meshes, final part meshes, and
compare to 0048a's original composition before increasing the resolution.

Tests:
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_sdf_puffer_sparse_groups_0049.py' -v

Next: 0050 defines strict root-leaf membership and UDF ownership verification.
