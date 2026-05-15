#!/bin/bash
#!/bin/bash
post='postprocess'
meshname='puffer_unitbb'
input='input'
output='output'
postseg=4
version=1


python3 reassign_npz.py \
  --in_npz "${post}/${output}/npz/${meshname}_${postseg}_postprocess_segments_v${version}.npz" \
  --ply_dir "${post}/${output}/segments/$meshname/${meshname}_${postseg}_v${version}" \
  --out_npz "${post}/${output}/npz/patched_${meshname}_${postseg}_postprocess_segments_v${version}.npz" \
  --out_dir "${post}/${output}/segments/$meshname/patched_${meshname}" \
