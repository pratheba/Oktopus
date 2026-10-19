Oktopus 0043 — Riggable root/leaves assembly for puffer
========================================================

Base: Install over 0042 (which includes 0041 + earlier patches).

New YAML: adapt_oktopus_puffer_root_leaves.yaml
  Copy to exp/train/manipulation/adapt_oktopus_puffer_root_leaves.yaml

YAML keeps combine_adaptations: true and the original blend_groups.
root_leaves:
  root: puffer_body_wrap_no_snug
  export_legacy_combined: false
  leaves:
    tentacle_t1: oktopus_9_v1|t1
    tentacle_t2: oktopus_9_v1|t2
    tentacle_t3: oktopus_9_v1|t3
    tentacle_t4: oktopus_9_v1|t4
    tentacle_t5: oktopus_9_v1|t5
    tentacle_t6: oktopus_9_v1|t6
    tentacle_t8: oktopus_9_v1|t8

HOW IT WORKS
- Root body is its own mesh object.
- Each tentacle is its own mesh object, even when leaves overlap.
- t1/t2/t4/t5 preserve their existing blend_groups and seam fitting within
  each logical tentacle. No blend_group is allowed to span different leaves.
- GLB assembly consists of 8 named geometries, also saved as 8 PLY meshes.
- With export_legacy_combined: false, no global min-union is extracted.
- The scene preserves separate mesh identities for rigging. Body-to-tentacle
  attachments are spatial overlaps, NOT watertight Boolean welds. A welded
  root with several leaf attachments cannot be split by connected components.
- Existing individual section exports are preserved if export_individual_meshes:
  true. The new rigging YAML sets export_individual_meshes: false to avoid
  re-extracting 15 intermediate sections. Set it true if a downstream UDF
  cutting pipeline still needs those individual section meshes.
- The original puffer YAML is not modified.

RUN / IMPORTANT FLAGS
Use the same puffer command and select:
   --test_file adapt_oktopus_puffer_root_leaves.yaml
   --surface-extraction mc
   --sdf-localize-after-gate
   --dcsdd-fast-bbox-marking
Do not use the 0040 rigid-only --dcsdd-stream-one-pass or crop-first flags.
Prefer a new output directory; start at 512 for rigging inspection.
If using --sdf-grid-memmap-dir, unused global union grid is skipped; active
intra-tentacle group buffers use the memmap scratch directory.

OUTPUTS
  <output>/<shape>/<shape>_adapt_riggable.glb
  <output>/<shape>/riggable_parts/body.ply
  <output>/<shape>/riggable_parts/tentacle_t{1,2,3,4,5,6,8}.ply
Other individual section exports unchanged.

To ALSO emit the old fused mesh, set root_leaves.export_legacy_combined:
true (that fused output is NOT riggable, since intersecting leaves Boolean-join).

VALIDATION
Unit tests cover YAML mapping, seam-group isolation, missing pieces,
GLB round-trip with intersecting leaves retained as independent objects,
and support-aware cropped MC geometry versus original masked MC.
The full Oktopus neural pipeline has NOT been benchmarked for this new mode.
