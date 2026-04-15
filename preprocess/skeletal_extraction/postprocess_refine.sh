#!/bin/bash
#!/bin/bash
post='postprocess'
meshname='boots_unitbb'
input='input'
output='output'
preseg=2
postseg=1
preversion=1
postversion=1


python3 postprocess.py \
  --in_npz "${post}/${output}/npz/${meshname}_${preseg}_postprocess_segments_v${preversion}.npz" \
  --ops_json "${post}/${input}/json/${meshname}_${postseg}_postprocess_v${postversion}.json" \
  --out_npz "${post}/${output}/npz/${meshname}_${postseg}_postprocess_segments_v${postversion}.npz" \
  --out_dir "${post}/${output}/${meshname}" \

python3 extract_segments.py \
   "${post}/${output}/npz/${meshname}_${postseg}_postprocess_segments_v${postversion}.npz" \
   "${meshname}_${postseg}_v${postversion}" \
   "${post}/${output}/${meshname}" 


#python3 postprocess.py \
#  --in_npz "${outfolder}/${meshname}/${meshname}_${numseg}_${refine}_postprocess_segments.npz" \
#  --ops_json "data/json/${meshname}_2_postprocess.json" \
#  --out_npz "${outfolder}/${meshname}/${meshname}_2_postprocess_segments.npz" \
#  --out_dir "${outfolder}/${meshname}" \
#
#python3 extract_segments.py \
#   "${outfolder}/${meshname}/${meshname}_2_postprocess_segments.npz" \
#   "${meshname}_2" \
#   "${outfolder}/${meshname}" 
