#!/usr/bin/env python3
"""Make root-level aliases because current Oktopus runners require flat YAML basenames.
Safe: refuses existing conflicts; doesn't overwrite user files. Dry-run by default.
"""
from pathlib import Path
import argparse
import os

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--apply', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parent
count = 0
for sub in ('sdf','sdf_udf'):
    for child in sorted((root/sub).glob('*.yaml')):
        dest = root/child.name
        if dest.is_symlink() and dest.resolve() == child.resolve():
            print('EXISTS (correct link):',dest.name)
            continue
        if dest.exists() or dest.is_symlink():
            print('SKIP (conflicting existing file):',dest.name)
            continue
        print('CREATE' if a.apply else 'WOULD CREATE',dest.name,'->',f'{sub}/{child.name}')
        if a.apply:
            os.symlink(f'{sub}/{child.name}',str(dest))
        count += 1
print('aliases:',count,'applied:',a.apply)
