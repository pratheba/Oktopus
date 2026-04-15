#!/bin/bash
#!/bin/bash
pre='preprocess'
infolder='input'
output='output'
post='postprocess'
meshname='oktopus_unitbb_300k'

python3 preprocess.py \
  --skel_file "${pre}/${infolder}/skel/${meshname}_skel-poly.polylines.txt" \
  --corr_file "${pre}/${infolder}/corr/${meshname}_correspondence-poly.polylines.txt" \
  --out_npz "${post}/${infolder}/npz/${meshname}_preprocess_segments.npz" \
  --out_json "${post}/${infolder}/json/${meshname}_preprocess_segments.json" \
  --n_keypoints 36 \
  --center_shift_scale 0.6 \
  --center_max_shift_frac 0.35 \
  --center_smooth_win 5 \
  --extend_min 0.01 \
  --extend_radius_alpha 2.5 \
  --extend_burial_alpha 1.2 \
  --extend_min 0.01 \
  --target_spacing 0.008

python3 extract_segments.py \
   "${post}/${infolder}/npz/${meshname}_preprocess_segments.npz" \
   "${meshname}" \
   "${post}/${output}/${meshname}" 

