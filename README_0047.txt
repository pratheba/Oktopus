Oktopus 0047: unified SDF evaluation, STAGE 1 (not root-leaf stitching)
=====================================================================
Apply after 0046 + the 0046 YAML crop-grid hotfix.  Install with:
  unzip -o oktopus-0047-unified-sdf-stage1-files.zip
Or use the Git patch against the 0046+hotfix code.

What is actually shared:
  - SDFStreamPiece: global voxel-index order, one-pass pointwise inference,
    GLOBAL near-surface support calibration, compact replay, SDF composition.
  - Both the independent rigid boot and 2-piece combined wrap cone use it.
  - Cone additionally records avatar source-s for exact post-SDF seam blend.
  - --sdf-stream-unified automatically selects a validated composition path.
  - The old flags work unchanged; leaving --sdf-stream-unified OFF keeps the
    legacy behavior. Geometry formulas, support clamp, seam blend, DCSDD C++,
    and UDF cut are untouched.

SUPPORTED WITH --sdf-stream-unified:
  1. One independent rigid adaptation, combine_adaptations: false,
     export_individual_meshes: true, accessory support clamp.
  2. Two DIRECT wrap adaptations in one post-SDF cone seam group,
     combine_adaptations: true, export_individual_meshes: false,
     wrap_angular_follow: 0 and the other 0046 strict safety conditions.
Both select DCSDD 'ours' at iso 0, and require writable temp/scratch.

NOT YET SUPPORTED by this opt-in route:
  mixed puffer with root_leaves, true active auto-snug correction,
  arbitrary query-coordinate seam warps, MC extraction, or multi-group assembly.
These configurations should continue using their existing validated routes.
A single topologically continuous puffer mesh with independently riggable
regions is a separate root/leaf attachment and connectivity problem, not a
safe side effect of sparse field union.

RUN: use the existing test_dc.py invocation and add --sdf-stream-unified
instead of the old --dcsdd-stream-inference/--dcsdd-stream-one-pass flags
or --sdf-stream-combined-wrap. Or use the two new YAMLs directly:
  -y adapt_oktopus_boots_t7_0047_unified.yaml
  -y adapt_oktopus_cone_0047_unified.yaml
Include --dcsdd-methods ours --surface-extraction ours,
--sdf-grid-memmap-dir <scratch> and --dcsdd-stream-replay-dir <scratch>.
For cone, use a new output directory, and do not supply rigid boot stream flags.

Reference 1024 Boot 1 / 0041: accepted=47,383,570; negative=377,659;
crop=199^3; active=68,696; V=68,696; F=138,456;
time=180.78 s.  Reference cone 512 / 0045 (also matched 0046 field):
evaluated=4,167,579; negative=386,655; crop=188^3;
active=73,581; V=73,581; F=147,312.
The 0046 sparse wrap code took 856.90 s vs 0045's 378.11 s.
The 0047 structural refactor does not yet claim to fix that performance
regression; it adds separate finish/compose/source-write timings to identify it.

TESTED locally:
 PYTHONPATH=src:src/ngc:tests python -m unittest -q \
   tests.test_sdf_stream_unified_0047 tests.test_sdf_combined_wrap_0046 \
   tests.test_dcsdd_stream_one_pass_2048 tests.test_dcsdd_stream_crop_2048 \
   tests.test_dcsdd_stream_replay_2048 tests.test_sdf_zero_angular_wrap_0045 \
   tests.test_sdf_wrap_after_gate_0042 tests.test_sdf_localize_after_gate_0041 \
   tests.test_dcsdd_bounded_filtering
