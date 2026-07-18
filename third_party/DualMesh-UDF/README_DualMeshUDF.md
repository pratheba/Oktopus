# DualMeshUDF (vendored)

Surface extraction from **unsigned** distance fields (ICCV 2023).
Paper: https://arxiv.org/abs/2309.08878 · Upstream: https://github.com/cong-yi/DualMesh-UDF

This is the known-good copy used by the Oktopus UDF inference/eval path
(`UDF/eval/gt_reconstruct.py` and the agent's UDF inference). It is a C++/pybind
extension and must be compiled once in your training/inference environment.

## Install (on the cluster)

```bash
cd third_party/DualMesh-UDF
unzip -o DualMesh-UDF-master.zip
cd DualMesh-UDF-master
# needs a C++ compiler + CMake; Eigen is bundled inside the package
pip install .
# also required by our GT reconstruction test:
pip install libigl trimesh
```

Verify:

```bash
python -c "from DualMeshUDF import extract_mesh; print('ok')"
```

## API we rely on

```python
from DualMeshUDF import extract_mesh
mesh_v, mesh_f = extract_mesh(udf_func, udf_grad_func, batch_size=150000, max_depth=7)
#   udf_func(pts)      -> (N,1) non-negative distances
#   udf_grad_func(pts) -> ((N,1) distances, (N,3) unit gradients)
#   extraction domain is the cube [-1,1]^3  (normalize meshes/queries into it)
```

`example/neural_utils.py` in the archive also provides `extract_mesh_from_udf(net, device)`
for a torch network whose forward maps (N,3) -> (N,1) UDF.
