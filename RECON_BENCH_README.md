# Reconstruction benchmarks: DC-SDD (SDF) and NSDUDF (UDF)

Two new, self-contained reconstruction back-ends were added. **No existing
file was modified** — everything is new files selected by their own runners.

| File | What it does |
|---|---|
| `src/app/agent_3dvec_sdf_dc.py` | `AgentSDFDC(AgentSDF)` — feeds the network's **SDF** grid into the *Dual Contouring of Signed Distance Data* `run_ours.py` suite: Marching Cubes, Dual Contouring, DC-SDD ("Ours"), Reach-for-the-Arcs, Kohlbrenner cones/RC. |
| `src/app/agent_3dvec_udf_nsdudf.py` | `AgentUDFNsdudf(AgentUDF)` — meshes the network's **UDF** with **NSDUDF** (neural pseudo-SDF + custom marching cubes) instead of DualMesh-UDF. |
| `inference_3dvec_dc.py` | Runner for the SDF suite. |
| `inference_3dvec_nsdudf.py` | Runner for the NSDUDF UDF path. |

Both back-ends reuse the entire existing inference path (model/handle
loading, curve localization, grid filling in `action_ngcnet_inference`) and
only replace the final meshing step, so results are directly comparable to
the pipeline's existing MC / RFTA / DualMesh-UDF outputs (same world frame,
no axis flip, no rescale).

---

## 1. SDF → `run_ours.py` suite (`inference_3dvec_dc.py`)

### How the format bridge works
`MCGrid` stores a flat `(N+1)^3` SDF array with `k_basis=[1,N+1,(N+1)^2]`
and world position `p = ijk*step + origin`, `origin=[-size]^3`,
`step=2*size/N`. The reference builds its grid with
`contouring.build_grid((n,n,n), min, max)` (= `igl::grid` scaled into the
cube), which yields the identical coordinate `-size + i*step`. The agent
therefore builds `U` at its own world extent `[-size, size]`, reindexes `S`
from `val_grid`, and calls:

```python
verts, faces = contouring.py_contouring(S, U, N1, N1, N1, isoValue, opts, None, None)
```

Contouring methods (MC/DC/Ours) get the full dense grid (sign-based, the
`+10` background is harmless). Point-based methods (RFTA, Kohlbrenner) get
only the **active** cells the network actually filled, because they read
`|S|` as a geometric radius.

### Build the reference once
```bash
cd ~/Downloads/dual-contouring-of-signed-distance-data-main
# submodules are needed for the externals; if you only cloned the zip, at least
# the core bindings must build:
python -m pip install -r requirements.txt
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
cd ..
python -c "import sys; sys.path.insert(0,'.'); import src.python.contouring as c; print('contouring OK')"
```
This produces `src/python/contouring/_contouring_cpp_module*.so`.

- Marching Cubes / Dual Contouring / Ours need only that build.
- **Reach-for-the-Arcs** needs `gpytoolbox` (`pip install gpytoolbox`).
- **Kohlbrenner cones (`mnm1`)** needs the `external/maximal-empty-spheres`
  CGAL build; **Kohlbrenner RC (`mnm2`)** needs `external/sdf-weighted-delaunay`
  built **and** a ground-truth mesh (`--dcsdd-gt-mesh`).

Any method whose external isn't built is logged and skipped — the rest still
run.

### Run
```bash
python inference_3dvec_dc.py \
    -c config/config_grid_b13d3_oktopus_dress1.yaml \
    -o dress1_recon -s dress1 -y test.txt -r 128 \
    --dcsdd-repo ~/Downloads/dual-contouring-of-signed-distance-data-main \
    --dcsdd-methods mc,dc,ours          # omit to run all six
```
Outputs: `inference/<num_samples>/dress1_recon/<shape>/<shape>_<method>_<ckpt>_mesh<res>.ply`
plus a per-method timing line, e.g.
`[dcsdd] timings for dress1 @ 128^3: mc=0.4s, dc=0.9s, ours=41.2s, rfta=88.0s`.

Point `DCSDD_REPO` via env instead of `--dcsdd-repo` if you prefer. DC-SDD
"Ours" parameters (`--dcsdd-outer-iters`, `--dcsdd-inner-iters`, `--dcsdd-mu`,
`--dcsdd-dc-weight`, `--dcsdd-batch-size`) match the paper defaults.

---

## 2. UDF → NSDUDF (`inference_3dvec_nsdudf.py`)

NSDUDF turns a UDF + gradient field into a signed **pseudo-SDF** on a grid
(one 8-value cell per voxel) and marches it. The agent's UDF oracle (the same
one the DualMesh-UDF path uses) returns distances in `[-1,1]` cube units and
unit gradients pointing away from the surface — exactly NSDUDF's expectation.

### Build NSDUDF once
```bash
# extract the uploaded nsdudf zip, e.g. to ~/Downloads/nsdudf-main
cd ~/Downloads/nsdudf-main
python -m pip install -r requirements_cuda_pip.txt    # or the conda file on macOS
cd custom_mc && python setup.py build_ext --inplace && cd ..
python -c "import sys; sys.path.insert(0,'custom_mc'); import _marching_cubes_lewiner; print('custom_mc OK')"
```
`model.pt` (the trained 32→128 classifier) ships in the repo root.

### Run
```bash
python inference_3dvec_nsdudf.py \
    -c config/config_udf_grid_b13d3_oktopus_puffer.yaml \
    -o puffer_nsdudf -s puffer -y test.txt -r 128 \
    --nsdudf-repo ~/Downloads/nsdudf-main --nsdudf-grid 129
```
Use `--nsdudf-grid 129` or `257` (power-of-2 + 1) if you later want the
relaxed DualMesh-UDF variant. `NSDUDF_REPO` / `NSDUDF_MODEL` env vars work as
alternatives to the flags. Domain flags (`--udf-domain-band`,
`--udf-domain-padding`, `--udf-far-value`, `--udf-batch-size`,
`--udf-cleanup`) behave exactly as in `inference_3dvec.py`.

---

## Notes
- Nothing here edits `inference_3dvec.py`, the agents, or `app/__init__.py`.
  To wire these into the existing `-a` selector later, that would be a
  one-line change per agent — left undone per project policy.
- Both back-ends depend on external C++/Cython builds that must be compiled on
  your machine; the code fails with an explicit, actionable message if a build
  or model file is missing.
