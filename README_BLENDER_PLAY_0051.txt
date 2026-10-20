Oktopus 0051 -- play the offline tentacle/boot animation in Blender
===================================================================
The standard skeleton NPZ is an INPUT to the 0051 pose generator.
The generator emits pose_000.glb..pose_008.glb (full static scene per pose),
boot_000.ply..boot_008.ply (deformed moving mesh),
skeleton_t7_000.npy..skeleton_t7_008.npy, and pose_manifest.json.

This Blender script turns the PLY frames into animated morph targets (shape
keys), imports static context from pose_000.glb, displays the changing curve
as a yellow wire, and saves one .blend with a playable timeline.

1. Generate pose files, e.g.:

  python tools/offline_tentacle_motion_0051.py \
    --handle-npz DATA/oktopus_9_v1/handle/std_handle.npz \
    --curve t7 --moving boot=INFERENCE/boot_512.ply \
    --static body=INFERENCE/riggable_parts/body.ply \
    --frames 9 --angle-deg 25 --axis z \
    --output inference/animation_t7_boot_512

2. Run Blender against the resulting folder, on Mac, Linux or headless HPC:

  blender -b --python tools/blender_play_0051.py -- \
      --pose-dir inference/animation_t7_boot_512 --step 6 --fps 24

3. Open `inference/animation_t7_boot_512/animation_0051.blend` in Blender,
   switch to the Layout workspace and press Spacebar on the timeline.
   It plays from rest to bend and back. Use the mouse to orbit the model.

Mac Blender path if the command is unavailable:
  /Applications/Blender.app/Contents/MacOS/Blender

The boot is the moving mesh; the body is static context. To also animate a
visible tentacle surface, give it as a second `--moving` mesh *bound to t7*.
Your puffer YAML currently has no t7 leaf. The t1 puffer leaf and t7 boot
cannot both use the same deformation curve without changing the adaptation.

For a blend-group demo instead:
  --curve t1 --moving puffer_t1=.../riggable_parts/tentacle_t1.ply \
  --static body=.../riggable_parts/body.ply

Notes:
- One .blend is a real, scrubbable animation; the original pose_NNN.glb files
  are NOT individual animations, only static scenes.
- This is precomputed offline playback, not live skeletal deformation or
  physics/rigging. Shape keys interpolate between the sampled poses.
- The tool requires Blender with native PLY/glTF import; ordinary Python
  cannot run the Blender script because bpy is required.
- The script rejects changing vertex/face order between pose PLYs.
- Current 0050 root/leaf mesh assembly is not stitched; the demo cannot
  guarantee no root seam gap, collision or mesh self-intersection.
- The viewer has not been tested in Blender in this environment.
