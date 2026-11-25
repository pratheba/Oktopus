#!/usr/bin/env python3
"""Validate direct-cut YAML/SDF adaptation pairs, no training or inference."""
from pathlib import Path
import json,hashlib,sys,yaml
ROOT=Path(__file__).resolve().parent
pairs=['trex_dress','armadillo_dress','trex_puffer','armadillo_puffer','trex_boots','armadillo_boots']

def fingerprint(items):
    payload=json.dumps([{k:v for k,v in item.items() if not str(k).startswith('_')} for item in items],sort_keys=True,separators=(',',':'),default=str).encode()
    return hashlib.sha256(payload).hexdigest()

for name in pairs:
    a=ROOT/'sdf/paired_direct'/f'nov24_adapt_{name}_sdf_for_direct_0107.yaml'
    b=ROOT/'sdf_udf'/f'nov24_adapt_{name}_sdf_udf_direct_0107.yaml'
    sa=yaml.safe_load(a.read_text()); ub=yaml.safe_load(b.read_text())
    assert sa['sdf_fused_exterior_0087'] is True
    assert sa['adaptations']==ub['adaptations'], name
    assert sa['root_leaves']==ub['root_leaves'], name
    assert sorted(sum((x['adaptation_indices'] for x in ub['root_leaves']['parts'].values()), []))==list(range(len(sa['adaptations']))), name
    assert len(set(str(x['target_key']) for x in sa['adaptations']))>=1
    print('PAIRED',name,'adaptations',len(sa['adaptations']),'fingerprint',fingerprint(sa['adaptations'])[:16])
print('YAML pairing validation passed (not a geometry/inference test).')
