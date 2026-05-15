#!/bin/bash
#!/bin/bash
post='postprocess'
meshname='puffer_unitbb'
input='input'
output='output'
postseg=4
version=1

python3 postprocess.py \
  --in_npz "${post}/${input}/npz/${meshname}_preprocess_segments.npz" \
  --ops_json "${post}/${input}/json/${meshname}_${postseg}_postprocess_v${version}.json" \
  --out_npz "${post}/${output}/npz/${meshname}_${postseg}_postprocess_segments_v${version}.npz" \
  --out_dir "${post}/${output}/segments/${meshname}" 

python3 extract_segments.py \
   "${post}/${output}/npz/${meshname}_${postseg}_postprocess_segments_v${version}.npz" \
   "${meshname}_${postseg}_v${version}" \
   "${post}/${output}/segments/${meshname}" 
