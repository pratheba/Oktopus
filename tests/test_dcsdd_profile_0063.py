"""0063 Python bridge profiling: opt-in, stage visibility, output parity."""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.modules.setdefault('agent_3dvec_sdf', types.SimpleNamespace(AgentSDF=object))
spec = importlib.util.spec_from_file_location(
    'agent_3dvec_sdf_dc_profile_test',
    Path(__file__).resolve().parents[1] / 'src/app/agent_3dvec_sdf_dc.py',
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Agent = module.AgentSDFDC


class MCGrid:
    def __init__(self, N=10):
        self.reso = N
        N1 = N + 1
        step = 2.0 / N
        i, j, k = np.indices((N1, N1, N1))
        sdf = np.sqrt(((i - N/2)*step)**2 + ((j - N/2)*step)**2
                      + ((k - N/2)*step)**2) - 0.5
        self.val_grid = sdf.ravel(order='F')
        self.empty_marks = np.zeros(len(self.val_grid), dtype=bool)
        self.grid_config = {
            'origin': np.array([-1.0, -1.0, -1.0]),
            'step': step, 'k_basis': np.array([1, N1, N1*N1]),
            'level': 0.0,
        }


class DCSDDProfileTest(unittest.TestCase):
    def test_profile_does_not_change_contour_result(self):
        vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        faces = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
        fake_contouring = types.SimpleNamespace(
            _contouring_cpp_module=object(),
            py_contouring=lambda *args: (vertices.copy(), faces.copy()),
        )
        agent = Agent()
        agent._load_dcsdd = lambda cfg: (fake_contouring, None, None)
        agent._ours_options = lambda cont, cfg: {'method': 'ours'}
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {'dcsdd_methods': ['ours'], 'output_folder': tmp,
                   'dcsdd_crop_grid': True,
                   'dcsdd_output_pattern': 'test.ply'}
            outputs = []
            for profile in (False, True):
                output = io.StringIO()
                with mock.patch.dict(os.environ, {'OKTOPUS_DCSDD_PROFILE_0063': str(int(profile))}):
                    with contextlib.redirect_stdout(output):
                        result = agent.run_all_reconstructions(MCGrid(), cfg, 'test')
                outputs.append((result['ours'], output.getvalue()))
            np.testing.assert_array_equal(outputs[0][0].vertices, outputs[1][0].vertices)
            np.testing.assert_array_equal(outputs[0][0].faces, outputs[1][0].faces)
            self.assertNotIn('[dcsdd_profile_0063]', outputs[0][1])
            for stage in ('grid_native_validation', 'grid_crop', 'grid_coordinates',
                          'grid_diagnostics', 'field_masks', 'field_clip',
                          'field_edt_and_roi', 'safe_field_total_inclusive',
                          'pybind_cpp_contouring', 'mesh_construct', 'mesh_export',
                          'mesh_quality_and_reporting', 'total_run_inclusive'):
                self.assertIn('stage='+stage+' ', outputs[1][1], stage)

    def test_config_override_disables_environment(self):
        with mock.patch.dict(os.environ, {'OKTOPUS_DCSDD_PROFILE_0063': '1'}):
            self.assertFalse(module._dcsdd_profile_enabled_0063({'dcsdd_profile_0063': False}))
            self.assertTrue(module._dcsdd_profile_enabled_0063({}))
            self.assertTrue(module._dcsdd_profile_enabled_0063({'dcsdd_profile_0063': True}))


if __name__ == '__main__':
    unittest.main()
