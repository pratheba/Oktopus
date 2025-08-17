#!/bin/sh

dir_name="$1"
arg_name="$2"
root="../../Pack50Dataset"
root_handle="${root}/${dir_name}/handle"

#python3 skeleton.py \
#	--name "$arg_name" \
#	--skel_file "../../Pack50Dataset/${dir_name}/${arg_name}-skel-poly.polylines.txt" \
#	--corres_file "../../Pack50Dataset/${dir_name}/${arg_name}-correspondence-poly.polylines.txt"\
#      	--output_folder "../../Pack50Dataset/${dir_name}/handle" &
python3 skeleton.py \
	--name "${arg_name}" \
	--skel_file "${root}/${dir_name}/${arg_name}-skel-poly.polylines.txt" \
	--corres_file "${root}/${dir_name}/${arg_name}-correspondence-poly.polylines.txt"\
       	--output_folder "${root_handle}" 
python3 dataformat.py \
	--corner_segments_file "${root_handle}/${arg_name}_cornersegment_with_addedpoints.npy" \
	--inner_segments_file "${root_handle}/${arg_name}_innersegments.npy" \
	--corner_segments_xyz_file "${root_handle}/${arg_name}_cornersegments_xyz.npy" \
	--inner_segments_xyz_file "${root_handle}/${arg_name}_innersegments_xyz.npy" \
	--radius_file "${root_handle}/${arg_name}_keypoint_radius.npy" \
        --output_folder "${root}/${dir_name}/train_data" \
	--filename "$arg_name"

