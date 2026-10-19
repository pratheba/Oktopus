0051 — Offline single-curve tentacle deformation proof of concept
================================================================
Side branch ONLY. No SDF/MC/DCSDD or 0049/0050 modifications. No torch.

Purpose
-------
Extract the original 512 SDF mesh ONCE, then produce 3–9 rest/bent/return
poses by moving one Oktopus skeleton curve and carrying bound mesh vertices
with it. This is NOT skeletal skinning / Blender rigging, nor does it run the
neural network for every frame.

Source geometry
---------------
Use data_root/oktopus_9_v1/handle/std_handle.npz (not the preliminary
preprocess_segments.npz if it only has segment_0 ... segment_12 names).
The tool reads that file's segments[*]['name'] and keypoints directly.
The exact rest-pose centerline must correspond to the meshes (same avatar,
scale, units, coordinate axes and pre-transform state).

Run t7 with a BOOT (use actual paths from your cluster):

  python tools/offline_tentacle_motion_0051.py \
      --handle-npz /PATH_TO_DATA/oktopus_9_v1/handle/std_handle.npz \
      --curve t7 \
      --moving boot=PATH_TO_BOOT_SDF_PLY \
      --static body=PATH_TO_PUFFER_BODY_PLY \
      --frames 9 --angle-deg 25 --axis z \
      --output inference/animation_t7_boot_512

Multiple meshes on the SAME curve move together:
  --moving boot_base=... --moving boot_detail=...

Run puffer's 3-section BLENDED tentacle t1 instead:

  python tools/offline_tentacle_motion_0051.py \
      --handle-npz /PATH_TO_DATA/oktopus_9_v1/handle/std_handle.npz \
      --curve t1 \
      --moving puffer_t1=PATH_TO_RIGGABLE_PARTS/tentacle_t1.ply \
      --static puffer_body=PATH_TO_RIGGABLE_PARTS/body.ply \
      --frames 9 --angle-deg 25 --axis z \
      --output inference/animation_t1_puffer_512

Important: boots target t7 but the supplied puffer adaptation YAML contains
NO puffer t7 leaf. Do not attach puffer t1 geometry to the t7 skeleton.
To demonstrate one puffer+boot leaf, one must first configure and generate
both assets on the SAME target curve, or animate t1 and t7 separately.

Outputs
-------
  pose_000.glb .. pose_008.glb    (static full scenes, 1 per pose)
  boot_000.ply .. boot_008.ply    (per-moving-part PLY)
  skeleton_t7_000.npy ...        (deformed skeletal centerline per pose)
  pose_manifest.json             (angles, mesh diagnostics and caveats)

Each GLB is ONE STATIC POSE, not a single playable animated GLB/MP4.
The frame files may be imported in Blender for a turntable or morph animation.
No PLY files are overwritten.

Math / behavior
---------------
- Resample rest curve by arc length and bind vertices once using closest
  sampled curve edge (approximate curve projection).
- Smoothly bend the curve by rotating its tangent vectors and integrating
  (approximately retains the curve length).
- Translate+rotate each bound mesh-vertex offset with its new curve point.
- The first and last frames are rest poses; the middle frame is full bend.
- The initial 12% of the curve is fixed by default. `--root-end auto`
  selects the endpoint closest to the handle's body skeleton.
- Face connectivity and vertex order are unchanged; this does NOT prove
  self-intersection freedom, sealed body-tentacle junctions or watertightness.
  The root/leaf interfaces in the current 0050 output remain unstitched.

Limitations
-----------
- Independent meshes can still collide or separate when bending.
- This is a curve-attached mesh warp, not a posed-SDF re-evaluation and
  not a demonstration of an analytically exact generalized-cylinder warp.
- If --root-end auto chooses incorrectly, specify --root-end start or end.
- If selected --axis is parallel to the root tangent, use another axis.
- For a valid paper figure, inspect frame-to-frame face degeneracies,
  local distortion and attachment continuity in addition to visual quality.
- The uploaded repo does not contain the final 512 PLY inference output or
  the inference std_handle.npz: test it on the cluster with those assets.

Tests
-----
  python -m unittest discover -s tests \
      -p test_offline_tentacle_motion_0051.py -v

Only required packages: numpy, scipy, trimesh.
