"""0051: compare correction-preserving staged inputs to legacy 0050a."""
import contextlib
import io
import tempfile
import unittest
from unittest import mock

import numpy as np

from test_puffer_bounded_wrap_0050a import PufferWrapStagingTests, wrap_radius_stub, localize_module


MODEL_KEYS = frozenset(('samples_local', 'coords', 'angles', 'radius', 'rho', 'rho_n'))


class PufferMinimalStagingTests(PufferWrapStagingTests):
    def test_all_required_inputs_match_wrap_with_avatar_cutting(self):
        args = {**self.args, 'dcsdd_bounded_filtering': True,
                '_sdf_puffer_bounded_wrap_staging': True,
                'dcsdd_filter_batch_size': 43}
        self._parity(args, avatar_inference=True)

    def test_rigid_blend_piece_needs_only_avatar_coords(self):
        args = {**self.args, 'rigid_radius': True, 'wrap_radius': False,
                'cut_avatar': False, 'use_soft_snug': False,
                'use_hard_avatar_clamp': False,
                'dcsdd_bounded_filtering': True, 'dcsdd_filter_batch_size': 43}
        self._parity(args, avatar_inference=False)

    def test_explicit_avatar_gate_still_preserves_full_avatar(self):
        args = {**self.args, 'rigid_radius': True, 'wrap_radius': False,
                'cut_avatar': False, 'detail_avatar_gate': True,
                'dcsdd_bounded_filtering': True, 'dcsdd_filter_batch_size': 43}
        self._parity(args, avatar_inference=True)

    def _parity(self, args, *, avatar_inference):
        with tempfile.TemporaryDirectory() as tmp:
            self.grid.memmap_dir = tmp
            stream = io.StringIO()
            with mock.patch.object(localize_module, 'interpolate_wrap_radius1', wrap_radius_stub), \
                 contextlib.redirect_stdout(stream):
                full_acc, full_av, full_idx, full_inside = self.curve.filter_grid_adapt(
                    self.grid, args)
                acc, av, idx, inside = self.curve.filter_grid_adapt(
                    self.grid, {**args, '_sdf_puffer_minimal_staging': True,
                                'adapt_debug_counts': True})
            np.testing.assert_array_equal(full_idx, idx)
            self.assertIsNone(inside)
            self.assertIsNotNone(full_inside)
            self.assertEqual(set(acc), MODEL_KEYS)
            self.assertEqual(set(av), MODEL_KEYS if avatar_inference else {'coords'})
            for key in acc:
                np.testing.assert_array_equal(full_acc[key], acc[key], err_msg=key)
                self.assertEqual(full_acc[key].dtype, acc[key].dtype)
            for key in av:
                np.testing.assert_array_equal(full_av[key], av[key], err_msg=key)
                self.assertEqual(full_av[key].dtype, av[key].dtype)
            self.assertIn('minimal_staging=True', stream.getvalue())
            self.assertIn('payload_reduction_pct=', stream.getvalue())
            # The model inputs and source coords preserve ordering and exact values.
            self.assertGreater(len(idx), 0)

    def test_rejects_callback_mode(self):
        args = {**self.args, 'dcsdd_bounded_filtering': True,
                '_sdf_puffer_bounded_wrap_staging': True,
                '_sdf_puffer_minimal_staging': True,
                '_dcsdd_stream_callback': lambda *args: None}
        with self.assertRaisesRegex(ValueError, 'staged direct'):
            self.curve.filter_grid_adapt(self.grid, args)

    def test_rejects_both_radius_mappings(self):
        args = {**self.args, 'dcsdd_bounded_filtering': True,
                '_sdf_puffer_bounded_wrap_staging': True,
                '_sdf_puffer_minimal_staging': True,
                'rigid_radius': True, 'wrap_radius': True}
        with self.assertRaises(ValueError):
            self.curve.filter_grid_adapt(self.grid, args)


if __name__ == '__main__':
    unittest.main()
