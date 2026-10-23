# 0103 · Mixed rigid/wrap direct-section streaming

One target curve, one output part, 2+ sections with a contiguous post-SDF
`overlap_crossfade` seam may now mix `rigid_radius` and `wrap_radius`.
Each section must have exactly one radius mode. Other streaming restrictions
remain (e.g. no per-section `cut_avatar`, no nonzero angular-follow wrap,
no active auto-snug, no multi-target / multi-part fusion).

## What changed

1. `sdf_stream_combined_wrap` validator permits *either* direct rigid or direct
   wrap per section; `curve_handle.filter_grid_adapt` permits the same modes in
   its guarded combined-stream callback. The original mapping for each section
   remains untouched.
2. The optional postblend Boolean can be evaluated on a **canonical avatar
   field** (`avatar_field_mode: canonical_world`). After accessory-section
   crossfading, process the final union of global voxel IDs in chunks.
   For each chunk, project world positions onto the **full unchanged target
   curve** (`outside=True`, no source interval/control transforms or cylinder
   rejection), run the original target SDF model once per voxel, and apply
   `max(blended_accessory_sdf, clearance - avatar_sdf)`.
   This prevents the differing source-interval projections from disagreeing
   at the same voxel. It is NOT the same geometry as averaging two per-section
   avatar SDF predictions; compare resulting meshes at a low resolution.
3. `avatar_field_mode: auto` (default) selects `canonical_world` for a mixed
   group, and `section_parity` for a wrap-only group to retain old 0093 output.
   The explicitly specified `section_parity` mode is rejected for mixed groups.
4. Canonical Boolean work is chunked and applied in-place; the extra per-piece
   avatar prediction files are skipped in that mode.

## Example usage

Set the modified YAML `adapt_file` to one of:

- `exp/train/manipulation/oct23_adapt_oktopus_cone_rigid_first_wrap_second_0103.yaml`
- `exp/train/manipulation/oct23_adapt_oktopus_cone_wrap_first_rigid_second_0103.yaml`

These differ ONLY in which section is rigid. They inherit the original
section source intervals, target intervals, scale, tilt and support settings.
Their `sdf_stream_export_sections_0093` flag is false to reduce peak memory.

Check log messages:

- `[sdf_stream_combined_wrap] ... avatar_field_mode= canonical_world`
- `[sdf_stream_combined_wrap] seam_blend= ...`
- `[sdf stream 0103] POSTBLEND_AVATAR_BOOLEAN applied mode= canonical_world`
- `[dcsdd:crop]` and the resulting crop side

## Limits

- This patch does **not** solve large dense DCSDD crop allocations at 2048.
- The canonical SDF uses a full-target projection, which intentionally differs
  from old section-specific interval/control projections; it is more coherent
  for a common avatar cut, but still needs GPU visual comparison.
- No relaxation of the original per-section parity tolerance is made.
- The standalone unit tests do not constitute an end-to-end model inference
  parity check. Start with 128 or 256 and compare before high-resolution runs.
