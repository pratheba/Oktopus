Oktopus 0054 — isolate moving t7 surface from touching neighbor (demo-only)
============================================================================
Install ON TOP of 0051 + 0052 + 0053. Apply either the patch or ZIP files.

Bug: The distance-only avatar vertex weights in 0052/0053 can include vertices
of a nearby different tentacle, especially when a neighboring limb approaches
t7. This produces the unwanted motion circled in the user screenshot.

Fix: Keep the existing distance + competing-curve + root smooth weights, then
build a SELECTION graph from original mesh triangle connectivity. The graph
excludes the protected root region. Only the candidate connected surface
branch containing t7's distal tip is allowed to move. Other candidate
components have their weights set exactly to zero.

No original vertex/face indices are changed. The boot still follows the
same t7 curve. Works with both --motion bend and --motion straighten.
This is heuristic mesh-based OWNERSHIP selection, not skinning, seam welding,
new SDF inference or collision checking. If a neighboring limb is *welded*
to t7 away from the held root, the selection cannot distinguish the two;
inspect the exported poses visually. Likewise holes in the candidate surface
may fragment t7's selected branch; inspect the number of components printed.

Usage: run exactly the same 0053 command you used. 0054 is ON by default for
--deform-avatar. You will now see:

  [avatar_topology_lock] components=... eligible=... selected=... rejected=...

Use a NEW --output folder and rerun the pose generator; existing pose_###.glb
and Blender shape-key animation files contain the old pre-0054 deformations.
If needed, disable this behavior for a diagnostic comparison by adding:

  --avatar-no-topology-lock

Example:

python tools/offline_tentacle_motion_0051.py \
  --handle-npz PATH_TO/std_handle.npz \
  --curve t7 \
  --deform-avatar octopus=PATH_TO/ORIGINAL_CONNECTED_OKTOPUS.ply \
  --moving boot=PATH_TO/BOOT_512.ply \
  --motion straighten \
  --root-hold 0.12 \
  --avatar-root-blend-end 0.30 \
  --frames 15 \
  --output inference/demo_t7_straighten_0054

Then rebuild playback (replace the pose directory):

blender -b --python tools/blender_play_0051.py -- \
  --pose-dir inference/demo_t7_straighten_0054 --fps 24 --step 6

Tests:

python -m unittest discover -s tests -p 'test_tentacle_topology_isolation_0054.py' -v
python -m unittest discover -s tests -p 'test_connected_avatar_motion_0052.py' -v
python -m unittest discover -s tests -p 'test_straighten_tentacle_motion_0053.py' -v
python -m unittest discover -s tests -p 'test_offline_tentacle_motion_0051.py' -v

Known limits: local triangle adjacency cannot split two limbs that have been
merged together away from their root, and the nearest-tip component heuristic
assumes the deformed tentacle surface reaches its skeletal tip. In those
situations use an explicit target vertex group or stricter topology isolation.

Preserve the camera and custom boot color from your previous Blender file
------------------------------------------------------------------------
Normally the 0051 playback generator starts from an empty Blender scene.
Instead, with the SAME mesh names, rest geometry and pose count as before,
load your existing colored/camera-framed .blend file and replace ONLY pose
shape-key coordinates:

blender -b PATH_TO_YOUR_EXISTING_COLORED.blend \
  --python tools/blender_refresh_pose_geometry_0054.py -- \
  --pose-dir inference/demo_t7_straighten_0054 \
  --output inference/demo_t7_straighten_0054/animation_preserve_camera_0054.blend

Open the NEW animation_preserve_camera_0054.blend and press Spacebar. The
material assignments, camera, lighting, and animation keyframes stay intact.
The script refuses to overwrite your original .blend. It also fails if the
original and new rest meshes, vertex order, or shape-key layout differ;
if you changed from 9 to 15 poses, generate a fresh Blender file instead.
This Blender helper passed Python syntax checks but has NOT been tested by
launching Blender here.
