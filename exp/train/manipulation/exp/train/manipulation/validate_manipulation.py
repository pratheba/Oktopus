#!/usr/bin/env python3
"""Lightweight static and optional handle-data validation for Oktopus adaptation YAMLs."""
import argparse, collections, pathlib, sys
import yaml

class NoDuplicateKeyLoader(yaml.SafeLoader):
    pass

def construct_mapping(loader,node,deep=False):
    out={}
    for key_node,value_node in node.value:
        key=loader.construct_object(key_node,deep=deep)
        if key in out:raise ValueError(f"duplicate YAML key: {key!r} (line {key_node.start_mark.line+1})")
        out[key]=loader.construct_object(value_node,deep=deep)
    return out
NoDuplicateKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,construct_mapping)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('directory',nargs='?',default='.')
    p.add_argument('--data-root',help='Optional Pack3MDataset root for std_handle.npz key checks')
    a=p.parse_args()
    root=pathlib.Path(a.directory)
    paths=sorted(root.glob('*.yaml'))
    failures=[];warnings=[]
    specs=[]
    for path in paths:
        try:
            obj=yaml.load(path.read_text(),Loader=NoDuplicateKeyLoader)
            arr=obj if isinstance(obj,list) else obj.get('adaptations',obj.get('adaptation')) if isinstance(obj,dict) else None
            if not isinstance(arr,list) or not arr: raise ValueError('no adaptation list')
            for i,d in enumerate(arr):
                if not isinstance(d,dict):raise ValueError(f'adaptation {i} not mapping')
                for k in ('target_key','accessory_key'):
                    if not isinstance(d.get(k),str) or d[k].count('|')!=1:
                        raise ValueError(f'adaptation {i}: missing/invalid {k}')
                if d.get('wrap_radius',False) and d.get('rigid_radius',False):
                    raise ValueError(f'adaptation {i}: conflicting wrap/rigid flags')
                if d.get('src_0')==d.get('src_1') or d.get('tgt_0')==d.get('tgt_1'):
                    raise ValueError(f'adaptation {i}: zero-length source/target interval')
            if isinstance(obj,dict):
                if obj.get('root_leaves') and not obj.get('combine_adaptations'):
                    raise ValueError('root_leaves requires combine_adaptations')
                if obj.get('dcsdd_stream_crop_first') and not obj.get('dcsdd_stream_inference'):
                    raise ValueError('crop_first requires stream_inference')
                if obj.get('dcsdd_stream_one_pass') and len(arr)!=1:
                    raise ValueError('one_pass requires exactly 1 adaptation')
            print(f'OK {path.name}: {len(arr)} adaptations')
            specs.append((path,arr))
        except Exception as e:
            failures.append(f'{path.name}: {e}')
    if a.data_root:
        try:import numpy as np
        except ImportError:
            sys.exit('numpy is required for --data-root')
        known={}
        for path,arr in specs:
            for rec in arr:
                for k in ('target_key','accessory_key'):
                    shape,curve=rec[k].split('|')
                    if shape not in known:
                        handle=pathlib.Path(a.data_root)/shape/'handle/std_handle.npz'
                        if not handle.is_file():
                            known[shape]=None
                            warnings.append(f'{shape}: missing {handle}')
                        else:
                            try:
                                with np.load(handle,allow_pickle=True) as blob:
                                    segs=blob['segments']
                                    known[shape]={str(x['name']) for x in segs}
                            except Exception as exc:
                                known[shape]=None
                                failures.append(f'{shape}: cannot inspect NPZ: {exc}')
                    if known[shape] is not None and curve not in known[shape]:
                        failures.append(f'{path.name}: missing {k}={rec[k]}, available={sorted(known[shape])}')
    print(f'\nFiles checked: {len(paths)}, failures: {len(failures)}, warnings: {len(warnings)}')
    if warnings:
        print('WARNINGS:');print('\n'.join('  '+x for x in sorted(set(warnings))))
    if failures:
        print('FAILURES:');print('\n'.join('  '+x for x in failures));return 1
    print('PASS: syntax/structure/optional handle names; NOT geometry or model quality')
    return 0
if __name__=='__main__':sys.exit(main())
