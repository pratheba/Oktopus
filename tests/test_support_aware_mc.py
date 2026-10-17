import numpy as np

from ngc.handle_utils.mc_utils import MCGrid


def _boundary_edges(mesh):
    if len(mesh.faces) == 0:
        return 0
    edges = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


def test_masked_mc_does_not_close_unknown_support_boundary():
    grid = MCGrid({"reso": 40, "size": 1.0, "level": 0.0})
    n1 = grid.reso + 1

    axis = np.linspace(-1.0, 1.0, n1)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    sdf = np.sqrt(x * x + y * y + z * z) - 0.70

    # Truncate sampled support through the sphere. Negative values reach the
    # support edge, reproducing the condition that makes plain MC connect them
    # to the +10 unknown/background sentinel.
    valid = x < 0.20
    flat_valid = valid.reshape(-1)
    grid.update_grid(
        sdf.reshape(-1)[flat_valid],
        np.flatnonzero(flat_valid),
        mark=True,
        mode="overwrite",
    )

    plain = grid.extract_mesh()
    masked = grid.extract_mesh_masked()

    assert len(plain.faces) > 0
    assert len(masked.faces) > 0
    # Plain MC invents a closed sheet at the unknown support boundary.
    assert _boundary_edges(plain) == 0
    # Masked MC leaves that edge open because there is no SDF evidence beyond
    # the sampled support.
    assert _boundary_edges(masked) > 0
    assert len(masked.faces) < len(plain.faces)
