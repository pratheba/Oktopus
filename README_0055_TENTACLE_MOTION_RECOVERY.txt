Oktopus 0055 — Restore moving t7 after overly restrictive 0054 lock
===================================================================
Apply ON TOP of 0054, on your animation side branch. This replaces only
  tools/offline_tentacle_motion_0051.py
and adds
  tests/test_tentacle_topology_recovery_0055.py
No changes to the bending/straightening equations, skeleton, boot geometry,
mesh triangles, Blender materials/camera, or production SDF pipeline.

Why 0054 could stop t7:
0054 selected one connected component whose *single vertex* was closest to
the target skeleton tip. If proximity/ownership gates divided the actual t7
surface into fragments, a small unrelated surface near the tip could win;
then nearly all t7 weights became zero.

0055 instead builds candidate connectivity with a relaxed geometric envelope
(still excludes the protected root) and ranks candidate components by the
number of skeletal-progress intervals they cover. Tie-breaking favors smaller
median distance to t7, not larger mesh tessellation. It can recover small
additional fragments if their progress intervals complement the main branch,
without including parallel neighboring tentacles that cover the same interval.
Original per-vertex motion weights are NOT increased. The new selection
merely decides which of the original candidate weights to retain.

Run the SAME 0054 --motion straighten (or bend) command, but write to a new
output directory, e.g. inference/demo_t7_straighten_0055. For the first run
watch these lines:

[avatar_topology_lock_0055] components=... selected_components=...
  eligible=... movable_before=... movable_after=...
  retained_ratio=... s_extent=(...)
[avatar_motion] octopus: .../... vertices follow selected curve

If movable_after is still unexpectedly small or the selected tentacle remains
stationary, send those lines and, ideally, the original rest avatar PLY and
std_handle.npz. Without the real mesh/skeleton this heuristic has only been
tested synthetically. A deterministic explicit t7 vertex mask is the next
fallback if several tentacles are topologically welded outside the root.

For a quick diagnostic to recover 0052/0053 behavior, append:
  --avatar-no-topology-lock
This restores motion but may also move the neighboring tentacle, as before.

To rebuild Blender playback (new pose frames):
  blender -b --python tools/blender_play_0051.py -- \
    --pose-dir inference/demo_t7_straighten_0055 --fps 24 --step 6

Or use the existing 0054 Blender geometry-refresh helper when preserving a
colored `.blend` with the same pose count and rest mesh. No need to reinstall
that helper: 0055 only modifies the pose-generator tool.

Tests (14 total):
  python -m unittest discover -s tests -p 'test_offline_tentacle_motion_0051.py' -q
  python -m unittest discover -s tests -p 'test_connected_avatar_motion_0052.py' -q
  python -m unittest discover -s tests -p 'test_straighten_tentacle_motion_0053.py' -q
  python -m unittest discover -s tests -p 'test_tentacle_topology_isolation_0054.py' -q
  python -m unittest discover -s tests -p 'test_tentacle_topology_recovery_0055.py' -q
