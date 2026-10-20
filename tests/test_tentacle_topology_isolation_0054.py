"""0054 regression: another tentacle must not follow t7 just because it is nearby."""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from offline_tentacle_motion_0051 import (sample_curve, bind_connected_avatar,
                                          bend_positions, straighten_positions)


def two_branches_sharing_fixed_root():
    """Two ribbon tentacles whose ONLY connection is in the first 12% of t7.

    The neighbor is so close to t7 that the old nearest-curve distance
    heuristic assigns it nonzero deformation weights (the user screenshot).
    """
    xs = np.linspace(0., 1., 21)
    vert, faces, t7_indices, neighbor_indices, common_indices = [], [], [], [], []
    for x in xs:
        j = len(vert)
        vert.extend([[x, -.035, 0.], [x, .035, 0.]])
        t7_indices.append([j, j+1])
    for i, x in enumerate(xs):
        if i <= 2:  # both tentacles share geometry only inside the fixed root
            neighbor_indices.append(t7_indices[i])
            common_indices.extend(t7_indices[i])
        else:
            j = len(vert)
            vert.extend([[x, .155, 0.], [x, .215, 0.]])
            neighbor_indices.append([j,j+1])

    def attach_ribbon(grid):
        for before, after in zip(grid[:-1], grid[1:]):
            faces.extend(([before[0], before[1], after[0]],
                          [before[1], after[1], after[0]]))
    attach_ribbon(t7_indices)
    attach_ribbon(neighbor_indices)
    return (np.asarray(vert, dtype=np.float64), np.asarray(faces, dtype=np.int64),
            np.asarray(t7_indices[7:]).ravel(), np.asarray(neighbor_indices[7:]).ravel(),
            np.asarray(common_indices))


class TopologyLockedTentacle0054(unittest.TestCase):
    def test_nearby_neighbor_is_not_deformed_despite_distance_gates(self):
        v, faces, target, neighbor, shared = two_branches_sharing_fixed_root()
        u, c = sample_curve(np.array([[0.,0.,0.],[1.,0.,0.]]), 128)
        # The available competitor curves are too far to distinguish the close
        # neighbor reliably. This matches the failure mode of distance gating.
        other = np.array([[0.,1.,0.], [1.,1.,0.]])
        kwargs = dict(distance_full=.10, distance_zero=.28, owner_band=.035,
                      root_hold=.12, root_blend_end=.30)
        _, _, unprotected = bind_connected_avatar(v, c, u, other,
                                                   topology_lock=False, **kwargs)
        self.assertGreater(np.max(unprotected[neighbor]), .10,
                           'Test must reproduce the original neighbor leakage')
        s, q, locked = bind_connected_avatar(v, c, u, other, faces=faces, **kwargs)
        self.assertTrue(np.all(locked[neighbor] == 0), 'Other tentacle must be locked')
        self.assertTrue(np.all(locked[shared] == 0), 'Common attachment stays fixed')
        self.assertGreater(np.max(locked[target]), .9)
        self.assertEqual(len(locked), len(v))
        for motion in ('bend', 'straighten'):
            if motion == 'bend':
                raw, _ = bend_positions(c, s, q, angle_deg=35,
                                         bend_axis=[0, 0, 1])
            else:
                # Use curved centerline for actual straightening case below.
                curved = np.column_stack((np.linspace(0,1,128),
                                          np.zeros(128),.2*np.linspace(0,1,128)**2))
                raw, _ = straighten_positions(curved,s,q,progress=1)
            posed = v + (raw-v)*locked[:,None]
            np.testing.assert_array_equal(posed[neighbor], v[neighbor])
            np.testing.assert_array_equal(posed[shared], v[shared])
            self.assertGreater(np.linalg.norm(posed[target]-v[target]), .1)

    def test_disabling_topology_lock_preserves_previous_behavior(self):
        v, f, _, _, _ = two_branches_sharing_fixed_root()
        u, c = sample_curve(np.array([[0.,0.,0.],[1.,0.,0.]]), 128)
        other = np.array([[0.,1.,0.],[1.,1.,0.]])
        a = bind_connected_avatar(v,c,u,other,faces=f,topology_lock=False)
        b = bind_connected_avatar(v,c,u,other)
        np.testing.assert_array_equal(a[2],b[2])


if __name__ == '__main__':
    unittest.main()
