#!/bin/bash
post='postprocess'
meshname='puffer_unitbb'
input='input'
output='output'
postseg=4
version=1

python3 update_radius.py \
  --in_npz "postprocess/output/npz/patched_${meshname}_${postseg}_postprocess_segments_v${version}.npz" \
  --out_npz "postprocess/output/handle_npz/${meshname}_${postseg}_postprocess_segments_v${version}.npz" \
  --out_dir "postprocess/output/handle_npz/${meshname}_${postseg}" \
  --version "${version}" \
  --q_wrap 0.999 \
  --mesh_path "preprocess/input/ply/${meshname}.ply"\
  --extra_surface_samples 200000 \
  --sample_keep_dist 0.03 \
  --smooth_keypoints_sigma 2.0 \
  --smooth_keypoints_preserve_endpoints \
  --resample_keypoints_by_arclen \
  --keypoint_arclen_ref 5.0 \
  --max_keypoints 64 \
  --min_keypoints 16 \
  --wrap_margin 0.1 \
  --wrap_relative_margin 0.1 \
  --wrap_w_factor 0.75 \
  --wrap_min_count 5 \
  --wrap_smooth_s 0.0 \
  --wrap_smooth_theta 2.0 \
  --cylinder_from_wrap \
  --derive_train_from_wrap \
  --cyl_margin 0.05 \
  --cyl_relative_margin 0.05 \
  --viz_ply
