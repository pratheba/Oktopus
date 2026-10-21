# 0074 — UDF evidence diagnostics on existing 0067 meshes

This is **not a new cutting algorithm**. It is an opt-in read-only audit of 0073's actual per-face UDF evidence. It does not call the UDF model again, rerun SDF, reintroduce avatar Booleans or intermediate meshes, or alter removal decisions.

## Why

The 0073 body report's `unowned_faces=80785` combines different failure modes. The new audit distinguishes:

- `not_assigned`: face outside every owning adaptation's curve interval (true interval miss).
- `assigned_no_raw_udf`: has owner interval, but no valid UDF face score.
- `raw_valid_abstained`: has a valid raw query, but saturation/metric-calibration rejected use of that owner.
- `effective_no_fused`: no valid combined score after ownership fusion (possible multi-owner mismatch).
- `below_grow`: usable score lower than grow threshold.
- `grow_only`: above grow threshold but below seed threshold.
- `seed_not_removed`: seed-strong face not in the final removed mask.
- `removed`: face actually selected for deletion (or proposed in 0073 dry-run mode).

For tentacles, the `face_evidence.npz` retains **separate per-adaptation** raw face scores, valid masks, calibrated thresholds (in the report), and ownership IDs. It does **not** falsely pretend these are the historical per-piece removed/kept surfaces.

## Install (on top of 0073)

```sh
cd /lustre/fast/fast/pselvaraju/Oktopus_now
git apply --check ~/Downloads/oktopus-0074-udf-evidence-diagnostics.patch
git apply ~/Downloads/oktopus-0074-udf-evidence-diagnostics.patch
```

Run with the same UDF model and final 0067 meshes, without `--mc-mesh`:

```sh
python test_sdf_udf_cut.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o puffer_0074_evidence_r512 \
  -s oktopus_9_v1 \
  -y udf_adapt_oktopus_puffer_0074_evidence_only.yaml \
  -r 512 -ckpt final \
  --udf-far-value 0.1 \
  --udf-model-batch-size 32768 \
  --udf-cut-query-batch-size 32768
```

The YAML differs from 0073 by just one enabled debug flag and annotations, and preserves all 15 adaptation dictionaries byte-for-byte to match the 0067 root/leaf manifest. Body remains cut under 0073 logic; seven tentacles remain dry-run-preserved.

## Outputs

For each part, under:

`inference/4M/puffer_0074_evidence_r512/oktopus_9_v1/udf_cut_evidence_0074/<part>/`

- `summary.json`: total, per-owner counts, 10 spatial slabs along each x/y/z axis, plus optional labelled regions.
- `face_evidence.npz`: original face IDs, centroids, raw UDF scores per adaptation (items 0..14), raw/effective validity, fused score, pre-topology and final removed masks.
- `not_assigned.ply`, `assigned_no_raw_udf.ply`, `raw_valid_abstained.ply`, `effective_no_fused.ply`, `below_grow.ply`, `grow_only.ply`, `seed_not_removed.ply`, `removed.ply` (only nonempty categories exported). These retain original world coordinates and can be loaded *alongside* the 0067 PLY in MeshLab.

`[udf cut evidence 0074] part=... unassigned=... no_raw=... abstained=... seed_not_removed=... removed=...` summarizes each part.

**Anatomical labels are not inferred automatically.** If known, add measured world-space neighborhoods to YAML:

```yaml
udf_cut_regions_0074:
- {part: body, name: sleeve, center: [X1, Y1, Z1], radius_world: R1}
- {part: body, name: neck, center: [X2, Y2, Z2], radius_world: R2}
- {part: body, name: bottom, center: [X3, Y3, Z3], radius_world: R3}
```

Coordinates are read from the unchanged 0067 mesh. This optional annotation makes the summary report coverage and cut decisions in those regions; without the coordinates, inspect the generic spatial atlas and PLY patches first.

## Verification

27 tests passed: 0074 unit/integration tests, 0073 metric, 0069 ownership, 0066 direct parts, and direct combined regression. A dedicated integration test confirms enabling diagnostics leaves kept/removed triangles and the fused scores **unchanged**. No real model/mesh inference was run in this environment.
