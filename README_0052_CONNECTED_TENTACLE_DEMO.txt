0052 - Connected full-avatar t7 + boot motion (offline 512 demo)
==============================================================
For the experimental side branch on TOP OF 0051 (not for main 2048).

The prior 0051 moved the boot because only the boot was supplied to --moving.
0052 adds --deform-avatar NAME=PLY to bind vertices of the ORIGINAL FULL
Oktopus avatar mesh to t7, while keeping the original face connectivity and
body attachment intact. Only vertices influenced by t7 move; nearby other
curves (including the body) suppress displacement. The root is locked during
the first 12% of normalized curve length, fading to full motion by 30%.
This is DEMO deformation, NOT skinning, Boolean geometry, collision simulation,
or a guarantee that rendered triangles do not self-intersect.

Required rest-pose inputs, ALL IN THE SAME COORDINATE FRAME:
 - DATA_ROOT/oktopus_9_v1/handle/std_handle.npz: Oktopus curve keypoints.
 - ORIGINAL_COMPLETE_OKTOPUS.ply: original full Oktopus avatar mesh with
   t7 attached to its body (prefer one connected mesh; do not use puffer body
   and boot alone, since that omits original t7 surface).
 - BOOT_AT_T7.ply: the adapted 512 boot from boots_1_v1|boot extraction.

Generate offline poses with both moving in the same operation:

python tools/offline_tentacle_motion_0051.py \
 --handle-npz DATA_ROOT/oktopus_9_v1/handle/std_handle.npz \
 --curve t7 \
 --deform-avatar octopus=ORIGINAL_COMPLETE_OKTOPUS.ply \
 --moving boot=BOOT_AT_T7.ply \
 --root-hold 0.12 --avatar-root-blend-end 0.30 \
 --frames 9 --angle-deg 25 --axis z \
 --output inference/demo_t7_octopus_boot_512

This produces:
  octopus_000.ply .. octopus_008.ply     (whole mesh, selectively deformed)
  boot_000.ply .. boot_008.ply           (same moving t7 centerline)
  skeleton_t7_000.npy .. _008.npy       (moving skeleton)
  pose_000.glb .. pose_008.glb          (scene poses)
  pose_manifest.json                   (moving = ["boot", "octopus"])

Then use the ALREADY-DELIVERED 0051 Blender playback script:

blender -b --python tools/blender_play_0051.py -- \
 --pose-dir inference/demo_t7_octopus_boot_512 --fps 24 --step 6

Open: inference/demo_t7_octopus_boot_512/animation_0051.blend
Press Space to play in Blender. Both octopus/t7 surface and boot animate.

If only the boot moves or the wrong body vertices move, inspect the log:
 [avatar_motion] octopus: N/T vertices follow selected curve
And tune world-unit gates in this order:
 --avatar-distance-full 0.15 --avatar-distance-zero 0.30
 --avatar-owner-band 0.035
 --root-hold 0.12 --avatar-root-blend-end 0.30
The default values assume approximately unit-normalized Oktopus coordinates;
if the original avatar is in a different scale or reference frame, convert
it first before running this script.

Do NOT supply the unstitched *_UNSTITCHED_parts.ply as the original octopus;
that is not a single connected body/tentacle mesh. The goal here is preserve
the pre-existing Oktopus mesh attachment and move its t7 vertices.

Validation:
  python -m unittest discover -s tests -p 'test_connected_avatar_motion_0052.py' -v
  python -m unittest discover -s tests -p 'test_offline_tentacle_motion_0051.py' -v

Both test modules pass on synthetic skeletons/meshes. This has NOT been
measured on the actual Oktopus avatar or Blender.
