#!/bin/bash
post='postprocess'
meshname='trex_unitbb'
input='input'
output='output'
postseg=9
version=1

python3 update_radius.py \
  --in_npz "postprocess/output/npz/patched_${meshname}_${postseg}_postprocess_segments_v${version}.npz" \
  --out_npz "postprocess/output/handle_npz/${meshname}_${postseg}_postprocess_segments_v${version}.npz" \
  --out_dir "postprocess/output/handle_npz/${meshname}_${postseg}" \
  --version "${version}" \
  --q_wrap 0.999 \
  --use_owned \
  --smooth_keypoints_sigma 2.0 \
  --smooth_keypoints_preserve_endpoints \
  --wrap_margin 0.05 \
  --wrap_relative_margin 0.02 \
  --wrap_w_factor 0.75 \
  --wrap_min_count 2 \
  --wrap_smooth_s 2.0 \
  --wrap_smooth_theta 3.0 \
  --cylinder_from_wrap \
  --derive_train_from_wrap \
  --cyl_margin 0.05 \
  --cyl_relative_margin 0.05 \
  --viz_ply
