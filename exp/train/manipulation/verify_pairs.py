#!/usr/bin/env python3
from pathlib import Path
import yaml
root = Path(__file__).resolve().parent
names = ['trex_dress', 'trex_puffer', 'trex_boots', 'armadillo_dress', 'armadillo_puffer', 'armadillo_boots']
for name in names:
    sdf = yaml.safe_load((root/'sdf'/f'nov25_{name}_sdf.yaml').read_text())
    udf = yaml.safe_load((root/'sdf_udf'/f'nov25_{name}_sdf_udf.yaml').read_text())
    assert sdf['adaptations'] == udf['adaptations'], name
    assert sdf['root_leaves'] == udf['root_leaves'], name
    assert sdf.get('sdf_fused_exterior_0087') is True, name
    part_owners = udf['root_leaves']['parts']
    flat = sum((list(p['adaptation_indices']) for p in part_owners.values()), [])
    assert sorted(flat) == list(range(len(sdf['adaptations']))), name
    print('OK',name,'adaptations',len(sdf['adaptations']),'parts',list(part_owners))
print('STRUCTURAL VALIDATION COMPLETE — not a geometry or checkpoint test.')
