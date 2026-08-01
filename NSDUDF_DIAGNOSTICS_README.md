# NSDUDF hole diagnostics for Oktopus

This bundle adds a separate diagnostic path. It does **not** replace or edit
`agent_3dvec_udf.py` or `agent_3dvec_udf_nsdudf.py`.

## Files

Copy these files into the Oktopus root:

```text
test_nsdudf_diag.py
diagnose_nsdudf_reference_mesh.py
src/app/agent_3dvec_udf_nsdudf_diag.py
src/app/nsdudf_diagnostics.py
```

## 1. Adapted Oktopus UDF diagnostic

Start with 64 NSDUDF cells. This is much faster and is enough to identify the
failure category.

```bash
python test_nsdudf_diag.py \
  -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
  -o diag_nsdudf_okto_puffer_64 \
  -s oktopus_9_v1 \
  -y adapt_oktopus_puffer_t1.yaml \
  -r 128 \
  -ckpt final \
  --nsdudf-repo /fast/pselvaraju/Oktopus_now/third_party/nsdudf \
  --nsdudf-grid 65 \
  --nsdudf-mesher marching_cubes \
  --nsdudf-oracle-chunk-size 16384 \
  --udf-batch-size 32768 \
  --nsdudf-diag-no-mesh
```

Important distinction:

```text
--nsdudf-grid 65   = 64 cells per axis
--nsdudf-grid 129  = 128 cells per axis
--nsdudf-grid 257  = 256 cells per axis
```

Remove `--nsdudf-diag-no-mesh` to also generate the pseudo-SDF mesh from the
same predictions used by the diagnostic.

Each adaptation item writes to:

```text
<output>/<shape>/nsdudf_diagnostics/item_00/
```

## 2. Exact mesh-UDF baseline

Run the same NSDUDF classifier using exact closest-point distances and
closest-point gradients from a reference mesh. The reference can be open.
Use the corresponding MC mesh or a ground-truth garment mesh, not the already
holey NSDUDF result.

```bash
python diagnose_nsdudf_reference_mesh.py \
  --mesh /path/to/reference_or_mc_mesh.ply \
  --nsdudf-repo /fast/pselvaraju/Oktopus_now/third_party/nsdudf \
  --grid 65 \
  --output exact_mesh_nsdudf_diag_64 \
  --no-mesh
```

This runner requires the `igl` Python module already used elsewhere in
Oktopus.

## Output files

`summary.json` contains all counts and percentiles.

The most useful fields are:

```text
cells.near_rejected_cells
cells.accepted_invalid_gradient_cells
cells.candidate_gradient_norm_percentiles
shared_faces.totals.active_active_disagreements
shared_faces.totals.surface_terminations
shared_faces.totals.surface_terminations_against_near_rejected
mesh.boundary_edges
mesh.nonmanifold_edges
```

Three point clouds show where failures occur:

```text
bad_faces_classifier_disagreement.ply   red
bad_faces_threshold_termination.ply     orange
bad_faces_other_termination.ply         yellow
```

Interpretation:

```text
Many orange points
    NSDUDF's 1.2× average / 2.0× maximum distance filter is rejecting
    cells where a neighboring prediction expects the surface to continue.

Many red points, exact baseline clean
    Oktopus UDF magnitudes or finite-difference gradients are outside the
    pretrained classifier's expected distribution.

Many red points in both exact and Oktopus runs
    This geometry is a difficult case for NSDUDF's independent per-cell
    classifier. The method has no cross-cell consistency guarantee.

High invalid-gradient count or gradient norms far from 1
    Finite-difference gradients are a major input problem.

Exact baseline has few bad faces but the Oktopus run has many
    The holes are not an unavoidable UDF limitation; they come from the
    learned/adapted UDF input.
```

## Threshold-only experiment

After the default diagnostic, test whether candidate rejection is the main
cause by changing only the threshold factors:

```bash
--nsdudf-diag-max-avg-factor 2.0 \
--nsdudf-diag-max-max-factor 3.0
```

Do not treat this as the final reconstruction setting yet. First compare
`surface_terminations_against_near_rejected` and the orange point cloud against
the default `1.2 / 2.0` run.
