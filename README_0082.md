# 0082 — Promote visually approved 0081 UDF meshes to the 8-part riggable assembly

This is an **assembly-only** step. It reads the *already generated* 0081
`udf_cut_curve_local_0081/<part>/proposed_kept_cleaned.ply` for each of the eight
puffer parts and uses Oktopus's established GLB + labeled PLY exporter.

- Does **not** run the UDF network, SDF adaptation or DCSDD.
- Does **not** modify geometry, smooth boundaries, Boolean union or weld attachments.
- Does **not** use `--mc-mesh` or per-adaptation intermediate meshes.
- Refuses partial/mismatched 0081 preview evidence and existing output folders.
- Creates **eight separately named riggable parts in one GLB**. This is an
  *unstitched assembly*, NOT a manifold mesh stitched at attachments.

## Apply (on top of 0081)

```bash
cd /lustre/fast/fast/pselvaraju/Oktopus_now
git apply --check ~/Downloads/oktopus-0082-promote-udf-assembly.patch
git apply ~/Downloads/oktopus-0082-promote-udf-assembly.patch
```

## Run once; no model inference

```bash
python tools/promote_udf_0081_assembly.py \
  --sdf-run-dir inference/4M/puffer_0067_final_parts_r512/oktopus_9_v1 \
  --udf-run-dir inference/4M/puffer_0081_curve_local_preview_r512/oktopus_9_v1 \
  --out inference/4M/puffer_0082_validated_cut_assembly_r512/oktopus_9_v1
```

The target **must not exist**; this avoids accidentally overwriting any of the
SDF/UDF inputs or a previous assembly. No YAML changes are needed.

## New outputs

```
inference/4M/puffer_0082_validated_cut_assembly_r512/oktopus_9_v1/
    riggable_parts_after_udf_cut/
        body.ply
        tentacle_t1.ply
        tentacle_t2.ply
        tentacle_t3.ply
        tentacle_t4.ply
        tentacle_t5.ply
        tentacle_t6.ply
        tentacle_t8.ply
    riggable_after_udf_cut.glb
    riggable_after_udf_cut_UNSTITCHED_parts.ply
    riggable_after_udf_cut_report.json
    root_leaves_ownership.json
```

The root-leaves ownership manifest retains the original 0067 adaptation
fingerprint and remaps face/vertex index ranges to the newly cut parts. The
labeled PLY contains `part_id` per face and vertex. The GLB retains eight
**separate** geometries with the corresponding original part names. These
meshes use the same world positions as their 0067 originals; no additional
rig transforms are applied.

## Validation

```bash
python - <<'PY'
import trimesh
p = 'inference/4M/puffer_0082_validated_cut_assembly_r512/oktopus_9_v1/riggable_after_udf_cut.glb'
s = trimesh.load(p, force='scene')
print('parts:', sorted(s.geometry))
print('faces:', {name: len(mesh.faces) for name, mesh in s.geometry.items()})
PY
```

For code tests (Python environment with project dependencies):

```bash
PYTHONPATH=src:src/app python -m pytest -q tests/test_promote_udf_0081_assembly_0082.py
```

**Not performed in 0082**: root-to-leaf watertight stitching, seam repair,
nonmanifold boundary repair, jagged rim smoothing, or cut-geometry changes.
Keep those as a separate validated phase after assembly; do not confuse a
single GLB containing eight geometries with a topologically welded mesh.
