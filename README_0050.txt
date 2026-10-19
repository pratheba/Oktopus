Oktopus 0050 — explicit puffer root/leaf ownership + UDF agreement
================================================================
INSTALL ON TOP OF 0049. Use replacement ZIP OR git patch, not both.

Implemented:
- Multi-section leaf selectors: blend_group t1/t2/t4/t5; single-section
  selectors for t3/t6/t8. Root is puffer_body_wrap_no_snug.
- Reuses the existing per-group seam optimization and 0049 cropped DCSDD
  individual/group extraction, without leaf/leaf global min-union.
- One static UNSTITCHED multi-part PLY and one GLB scene. PLY explicitly
  includes per-vertex AND per-face integer part_id fields; GLB has named
  body/tentacle mesh objects; manifest maps adaptation indices to parts.
- Both SDF and UDF YAMLs now have exactly matching 15 adaptation settings.
  The historical UDF-only tloc -0.05 for t4_2/t5_2 was removed to match SDF.
- The UDF stage requires the corresponding SDF ownership manifest. If the
  transforms or ownership differ, it refuses to apply cuts.
- UDF removes faces ONLY from the owning body/tentacle; no leaf/leaf merge.
- Unsupported 'single_connected' output fails explicitly.

IMPORTANT: This is an ownership/mesh-assembly foundation, NOT a validated
welded root-to-leaf junction. The one-file *_UNSTITCHED_parts.ply can be a
multi-component and/or intersecting surface. It does not satisfy the desired
final single connected watertight puffer. Actual body/tentacle seam trimming,
stitching and manifold verification still require another geometry patch.
No skeletal animation, skinning or pose deformation is added.

Validation run at 512:
  Existing test_dc.py DCSDD ours command, with
  -y adapt_oktopus_puffer_root_leaves_0050.yaml -r 512
  --surface-extraction ours --dcsdd-methods ours
  --sdf-grid-memmap-dir /fast/pselvaraju/Oktopus_now/logs

UDF cut after SDF stage:
  -y adapt_oktopus_puffer_blend_udf_cut_root_leaves_0050.yaml
  Before UDF cut, SET udf_cut_root_leaves_mesh_dir to the actual
  <SDF output>/oktopus_9_v1/riggable_parts location. Keep the per-section
  --mc-mesh references pointing to the corresponding SDF individual meshes.
  The UDF stage reads root_leaves_ownership.json from the parts directory's
  parent; it rejects any source/adaptation mismatch.

Tests:
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_root_leaf_ownership_0050.py' -v
  PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
      -p 'test_sdf_puffer_sparse_groups_0049.py' -v

Performance/scale limitations:
  Neither 0049 nor 0050 implements count-free streaming of all original
  puffer avatar-carved corrections. Bounded localization still stages many
  fields and may be expensive at 2048. Profile candidate counts, scratch,
  RSS and resulting geometry at 512, then 1024 before any 2048 run.
