"""Standalone parity tests for the Oct-18 DCSDD bridge optimization.

Run: python -m unittest discover -s tests -p test_dcsdd_bridge_memory.py
Does not depend on GPU/model inference/DCSDD bindings.
"""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

# Import without pulling in torch/training dependencies.
sys.modules.setdefault('agent_3dvec_sdf', types.SimpleNamespace(AgentSDF=object))
_p = Path(__file__).resolve().parents[1] / 'src/app/agent_3dvec_sdf_dc.py'
_spec = importlib.util.spec_from_file_location('agent_3dvec_sdf_dc_bridge_test', _p)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Agent = _mod.AgentSDFDC

class MCGrid:
    def __init__(self, N, values, empty):
        self.reso = N
        self.val_grid = values
        self.empty_marks = empty
        self.grid_config = dict(origin=np.array([-1.2, -1.2, -1.2]), step=2.4/N,
                                k_basis=np.array([1, N+1, (N+1)**2]), level=0.0)

    def idx2pts(self, inds):
        N1 = self.reso+1
        i=inds % N1
        j=(inds // N1) % N1
        k=inds // (N1*N1)
        return np.stack([i,j,k],axis=-1)*self.grid_config['step']+self.grid_config['origin']


def reference_prepare(values, empty, N1, step, level, cfg):
    field = np.asarray(values, dtype=np.float64).reshape(-1) - float(level)
    evaluated = np.isfinite(field) & ~np.asarray(empty, dtype=bool).reshape(-1)
    G = field.reshape((N1,N1,N1),order='F')
    E = (~evaluated).reshape((N1,N1,N1),order='F')
    inside = (~E) & (G < 0)
    dist = distance_transform_edt(~inside,sampling=(float(step),)*3)
    safe = G.copy()
    safe[E] = dist[E]
    band = cfg.get('dcsdd_sdf_band',4*np.sqrt(3)*step)
    return np.ascontiguousarray(np.clip(safe, -band, band).reshape(-1,order='F'))

class BridgeParityTest(unittest.TestCase):
    def test_grid_native_exact(self):
        for N in (2, 5, 13):
            rng=np.random.default_rng(N)
            n=(N+1)**3
            grid=MCGrid(N,rng.uniform(-.4,.4,n),rng.choice([False,True],n))
            x=Agent()
            legacy = x._grid_to_S_U_legacy(grid)
            fast = x._grid_to_S_U(grid)
            for a,b in zip(legacy[:3],fast[:3]):
                np.testing.assert_array_equal(a,b)
            self.assertEqual(legacy[3],fast[3])

    def test_clipped_edt_exact(self):
        for N in (5, 11, 24, 48):
            rng=np.random.default_rng(N+300)
            n=(N+1)**3
            step=2.4/N
            values=rng.uniform(-1,1,n)
            empty=rng.random(n)<.70
            for level in (0,0.18):
                for band in (4*np.sqrt(3)*step, 9*np.sqrt(3)*step):
                    cfg={'dcsdd_sdf_band':band}
                    expected=reference_prepare(values,empty,N+1,step,level,cfg)
                    actual=Agent._prepare_dcsdd_field(values,empty,N+1,step,level,cfg)
                    np.testing.assert_array_equal(actual,expected)

    def test_local_edt_matches_full_at_domain_boundaries(self):
        for N, center in ((35,(4,7,8)), (64,(30,38,27)),(51,(48,48,4))):
            N1=N+1
            step=2.4/N
            ix,jy,kz=np.ogrid[:N1,:N1,:N1]
            d=(ix-center[0])**2+(jy-center[1])**2+(kz-center[2])**2
            G=(np.sqrt(d)-4.0)*step
            E=np.ones((N1,N1,N1),dtype=bool)
            E[max(0,center[0]-8):min(N1,center[0]+9),
              max(0,center[1]-8):min(N1,center[1]+9),
              max(0,center[2]-8):min(N1,center[2]+9)]=False
            flat=G.ravel(order='F')
            marks=E.ravel(order='F')
            cfg={}
            expected=reference_prepare(flat,marks,N1,step,0.,cfg)
            actual=Agent._prepare_dcsdd_field(flat,marks,N1,step,0.,cfg)
            np.testing.assert_array_equal(actual,expected)

    def test_cropped_field_zero_crossings(self):
        # ROI margin keeps all original zero-crossing edges and their neighboring cells.
        for N in (20,48):
            N1=N+1
            step=2.4/N
            x = np.arange(N1)[:,None,None]
            y = np.arange(N1)[None,:,None]
            z = np.arange(N1)[None,None,:]
            radius=((x-0.6*N)**2 + (y-0.4*N)**2 + (z-0.5*N)**2)**0.5
            G=np.broadcast_to(radius-0.12*N,(N1,N1,N1)).copy()
            empty=np.zeros_like(G,dtype=bool)
            grid=MCGrid(N,G.ravel(order='F'),empty.ravel(order='F'))
            x=Agent()
            S,U,E,R=x._grid_to_S_U(grid,{'dcsdd_crop_grid':True},level=0)
            self.assertLessEqual(R,N1)
            self.assertEqual(len(S),R**3)
            self.assertEqual(U.shape,(R**3,3))
            # Cropped cube coordinates reproduce the native volume at every row.
            origin=grid.grid_config['origin']
            starts=np.rint((U[0]-origin)/step).astype(int)
            expected=G[starts[0]:starts[0]+R,starts[1]:starts[1]+R,starts[2]:starts[2]+R]
            np.testing.assert_array_equal(S,expected.ravel(order='F'))
            self.assertGreater(np.min(S),-np.inf)

if __name__=='__main__': unittest.main()
