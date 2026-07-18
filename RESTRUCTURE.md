# Oktopus restructure (SDF cleanup)

Branch: `restructure` (off `main`). Work in progress toward separating the
SDF pipeline from shared code, ahead of adding a UDF pipeline.

## New layout

```
src/                     # SHARED building blocks (used by SDF and, later, UDF)
├── ngc/                 # handle machinery: handle_3dvec, curve_handle,
│                        #   PWLA_curve_handle, handle_utils/, curve_functions/,
│                        #   curve_utils/, preprocess/ (skeleton extraction)
├── network/             # model_withgrid + mlp/mask/triplane/encodings
├── data/                # dataset_part
└── utils/               # mc_utils, options_3dvec, dotdict, vis_blender/

SDF/
├── preprocess/          # process_data_3dvec_closed_v2 + sdf_closure_utils_v2
│                        #   + close_all_boundaries + smooth_mesh
├── training/            # train_3dvec, loss_3dvec, utils
└── app/                 # agent_3dvec inference + app_utils_3dvec, blend/mix

UDF/
└── preprocess/          # process_data_3dvec_udf_keep_cylinder, check_udf_gt
                         #   (WIP: still needs udf_open_mesh_utils)

_archive/                # ~40 dead/dated snapshots (kept, out of the way)

# root: entry scripts (train_net*.py, application.py, inference*.py, test.py),
#       config/, exp/, preprocess/ (standalone skeletal-extraction tooling)
```

## How imports work after the move

The code uses a flat-import + `sys.path.append` style. Whole package
directories were moved intact, so all *internal* imports are unchanged.
Cross-package resolution is handled by a small path bootstrap prepended to
each entry script, which puts `src/`, `SDF/`, `UDF/` on `sys.path`:

    # --- project path bootstrap (restructure: src/, SDF/, UDF/) ---
    import os as _os, sys as _sys
    _ROOT = _os.path.dirname(_os.path.abspath(__file__))
    for _p in ('src', 'SDF', 'UDF'):
        _sys.path.insert(0, _os.path.join(_ROOT, _p))
    # --- end bootstrap ---

Entry scripts patched: train_net.py, train_net_3dvec.py, application.py,
inference.py, inference_3dvec.py, test.py, and the two moved preprocess
scripts (with a 2-level-deep variant of the bootstrap).

Only two import statements were rewritten (both moved out of `ngc`):
`from handle_3dvec import Handle` -> `from ngc import Handle`
(in SDF/preprocess and UDF/preprocess process_data scripts).

## Verified

- All 8 edited scripts compile.
- Static import scan of 100 live files: every project import resolves.
- The `ngc/curve_utils.py` vs `ngc/curve_utils/` name collision is resolved
  (stray module archived; the package was already the one winning).

## Pre-existing issues found (NOT caused by the restructure)

- Syntax errors in files that are NOT in the live pipeline:
  `src/ngc/curve_functions/_localize.py` (unclosed call ~line 61),
  `src/ngc/preprocess/skeleton_utils/_compute_radius.py`,
  `preprocess/build_cylinder.py`, `build_ring.py`, `export_segment.py`,
  `mesh_export.py`. `curve_functions/__init__.py` only imports `_interpolate`,
  so `_localize.py` is not loaded by the pipeline.

## Phase 3 — PWLA_curve_handle split (done, working-tree)

`src/ngc/PWLA_curve_handle.py` (one 4,613-line `PWLACurve` class, 90 methods)
was split into a `src/ngc/pwla_curve/` package of themed mixin classes:

    pwla_curve/_core.py         (_CoreMixin)        10 methods  construction/state
    pwla_curve/_frames.py       (_FramesMixin)      11 methods  frames/tangents/rotation
    pwla_curve/_radius.py       (_RadiusMixin)       8 methods  radius
    pwla_curve/_wrap.py         (_WrapMixin)        13 methods  wrap/envelope fields
    pwla_curve/_sdf.py          (_SDFMixin)          6 methods  SDF/implicit/projection
    pwla_curve/_localize.py     (_LocalizeMixin)    19 methods  localization
    pwla_curve/_stretch.py      (_StretchMixin)      7 methods  stretch ops
    pwla_curve/_interpolate.py  (_InterpolateMixin) 15 methods  interpolation/output

`PWLA_curve_handle.py` is now a thin assembler:
`class PWLACurve(_CoreMixin, _FramesMixin, ... , _InterpolateMixin): ...`
so `from PWLA_curve_handle import PWLACurve` is unchanged. The original
monolith is preserved at `_archive/ngc/PWLA_curve_handle_monolith.py`.

Verified: loss-less line partition (4585==4585), all 90 method defs preserved
(89 unique + the pre-existing duplicate `inverse_transform` kept together in
`_interpolate` so the 2nd still overrides), all 10 files compile, all imports
resolve. Behaviour-preserving refactor (mixins share `self`); still needs a
runtime pass in your torch env to be 100% confirmed.

## Next steps

1. Verify at runtime: run your training command; confirm imports resolve.
3. Build out UDF/ (add udf_open_mesh_utils, wire UDF loss/training).
