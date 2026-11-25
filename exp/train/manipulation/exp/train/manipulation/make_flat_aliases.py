#!/usr/bin/env python3
"""Expose nested YAMLs as root-level symlinks for legacy runners.

The inspected 0107 tools/udf_run_direct.py rejects --yaml containing a slash;
this creates basename aliases without duplicating or overwriting contents.
Run from anywhere. Without --apply, only reports proposed aliases.
"""
from pathlib import Path
import argparse
import os

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--category', choices=['sdf_udf','sdf','both'], default='sdf_udf')
    args=ap.parse_args()
    dirs=[]
    if args.category in ('sdf_udf','both'): dirs.append(ROOT/'sdf_udf')
    if args.category in ('sdf','both'): dirs.extend([ROOT/'sdf',ROOT/'sdf/paired_direct'])
    for folder in dirs:
        for src in sorted(folder.glob('*.yaml')):
            dst=ROOT/src.name
            expected=os.path.relpath(src,dst.parent)
            if dst.is_symlink() and os.readlink(dst)==expected:
                print('OK',dst.name, '->', expected); continue
            if dst.exists() or dst.is_symlink():
                print('SKIP existing',dst, '(not overwritten)'); continue
            print(('LINK' if args.apply else 'WOULD LINK'),dst.name,'->',expected)
            if args.apply: dst.symlink_to(expected)

if __name__=='__main__': main()
