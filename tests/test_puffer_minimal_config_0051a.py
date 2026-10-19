"""0051a: YAML option must reach extraction_config, not be silently dropped."""
import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/app/agent_3dvec_sdf.py'
YAML = ROOT / 'exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0050.yaml'


class PufferMinimalConfigForwardingTests(unittest.TestCase):
    def test_yaml_opt_in_is_allowed_through_top_level_loader(self):
        source_tree = ast.parse(SOURCE.read_text())
        options = []
        for node in ast.walk(source_tree):
            if not isinstance(node, ast.Compare):
                continue
            if not (isinstance(node.left, ast.Name) and node.left.id == 'key_text'):
                continue
            for rhs in node.comparators:
                if isinstance(rhs, (ast.Set, ast.Tuple)):
                    names = {v.value for v in rhs.elts if isinstance(v, ast.Constant)
                             and isinstance(v.value, str)}
                    if 'sdf_puffer_sparse_groups' in names:
                        options.append(names)
        self.assertEqual(len(options), 1, 'find exactly one top-level YAML forwarding list')
        self.assertIn('sdf_puffer_minimal_staging', options[0])
        self.assertIn('sdf_puffer_minimal_staging: true', YAML.read_text())

    def test_opt_in_reaches_puffer_guard(self):
        source = SOURCE.read_text()
        self.assertIn("extraction_config.get('sdf_puffer_minimal_staging', False)", source)
        self.assertIn("adapt_arg['_sdf_puffer_minimal_staging'] = bool(", source)


if __name__ == '__main__':
    unittest.main()
