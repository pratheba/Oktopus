# SDF preprocessor: reuse already-closed meshes (no recapping)

Use the **`--use_existing_closed_meshes`** option to prevent any calls to
`close_all_boundaries` and to prevent rewriting existing closed PLYs.

Required per dataset item (e.g. `Pack3MDataset/dress_2_v2`):

```text
mesh.ply
mesh_base.ply
full_parts/<original open per-curve meshes>
base_parts/<original open per-curve meshes>
closed_sdf_meshes/
  parts_full/000_<curve>_closed.ply
  parts_base/000_<curve>_closed.ply
  # optional: *_closed_caps_only.ply next to each closed part
  # optional: global/mesh_full_closed.ply, global/mesh_base_closed.ply
```

Original `full_parts`/`base_parts` meshes are used for source areas and
training samples from `handle/std_handle.npz`; do not overwrite them with the
closed parts.  Global files are reused from the closed cache if present;
otherwise the original global files must already be valid watertight volumes.

The preclosed part names may use `0_<curve>_closed.ply` instead of `000_...`.
An unambiguous ID-prefix fallback is also supported. Supply another closed
root using `--closed_mesh_dir` (absolute, or relative to each item).

Example (on the cluster, retaining your original training settings):

```bash
python3 SDF/preprocess/process_data_3dvec_closed_v2.py \
  -r 'Pack3MDataset/' -d 'data_dress2_2.txt' \
  -ns '[0.04, 0.08, 0.1, 0.3]' \
  -ss 3000000 -sv 400000 -k 12 -w \
  --closed_mesh_dir closed_sdf_meshes \
  --use_existing_closed_meshes
```

This will **fail** if any expected preclosed part is missing or invalid. It will
not attempt any boundary repair/capping or modify the closed mesh files.

## Important: artificial-cap supervision

SDF distance queries use the existing **closed** geometry even if cap-only
meshes are absent. However, the original training pipeline adds extra
on-surface/perturbed samples **only from** `*_closed_caps_only.ply`. If these
files are missing for an open source part, the code prints a warning and no
explicit cap samples are generated for that part. This changes the training
sample distribution; it is not equivalent to an existing successful run that
included cap-only supervision. Provide your previously generated cap-only
meshes if available. The cap-density flags adjust sampling and do not close
meshes.

Validation checks watertightness, winding consistency and valid oriented
volume. No model training/GPU inference is performed by the patch tests.
