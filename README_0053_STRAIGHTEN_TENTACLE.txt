Oktopus 0053 — offline straighten/rebend one tentacle and its boot
================================================================
INSTALL ON TOP OF 0051 + 0052 (animation side branch; unrelated to 2048 work).

Adds an optional `--motion straighten` to the existing
`tools/offline_tentacle_motion_0051.py`. `--motion bend` remains the default
and reproduces the existing 0052 behavior.

What happens:
  - Bind the original connected octopus mesh near t7 to its rest curve.
  - Hold the first --root-hold 0.12 of the curve exactly in place.
  - Rotate successive curve tangents toward the outgoing root direction,
    using a smooth spatial transition; integrate segments to preserve arc
    length. The segment directions beyond --full-bend 0.85 become straight
    when progress reaches 1. The held first 12% may remain curved.
  - Move the selected mesh vertices and the boot using the SAME curve motion.
    The full mesh keeps its existing faces and vertex indices.
  - Export rest -> straight -> rest pose_NNN.glb, octopus_NNN.ply,
    boot_NNN.ply, skeleton_t7_NNN.npy, pose_manifest.json.
  - No SDF reevaluation, bone painting or topology changes.

Run in the repository root (provide actual existing rest-pose paths):

python tools/offline_tentacle_motion_0051.py \
  --handle-npz DATA_ROOT/oktopus_9_v1/handle/std_handle.npz \
  --curve t7 \
  --deform-avatar octopus=/PATH/TO/ORIGINAL_CONNECTED_OKTOPUS.ply \
  --moving boot=/PATH/TO/ADAPTED_BOOT_512.ply \
  --motion straighten \
  --root-hold 0.12 \
  --avatar-root-blend-end 0.30 \
  --full-bend 0.85 \
  --frames 15 \
  --output inference/demo_t7_straighten_512

The --axis and --angle-deg options control *bend* only. They are ignored
in straighten mode, which uses the root's outgoing tangent.

In Blender, build pose playback with the existing 0051 Blender tool:

blender -b --python tools/blender_play_0051.py -- \
  --pose-dir inference/demo_t7_straighten_512 --fps 24 --step 6

Then open inference/demo_t7_straighten_512/animation_0051.blend and press
Spacebar to play. This creates a NEW .blend, not an edit to the already
colored/camera-framed bend-animation .blend. Reapply color/camera if needed;
do not overwrite your manually edited .blend.

The straightening is demo-only: topology is preserved but no collision-free,
nonintersecting or watertight guarantee is provided. Using the connected
avatar rest mesh, not the unstitched puffer parts, is important.

Tests:
  python -m unittest discover -s tests -p 'test_straighten_tentacle_motion_0053.py' -v
  python -m unittest discover -s tests -p 'test_connected_avatar_motion_0052.py' -v
  python -m unittest discover -s tests -p 'test_offline_tentacle_motion_0051.py' -v
