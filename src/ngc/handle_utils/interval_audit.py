"""0048: bounded audit of conservatively narrowed broad-phase candidates.

A narrower segment AABB union is only safe when all voxels accepted by the
UNCHANGED exact localizer are still in that union.  This checks precisely the
old-minus-new candidates without buffering the full grid or invoking the model.
"""
from time import perf_counter
import numpy as np


def audit_candidate_subset(full_chunks, narrow_chunks, idx2pts, localize,
                           *, stop_on_missing=True):
    """Audit sorted, unique flat-index chunk iterators; return audit diagnostics.

    `localize(points)` returns local indices accepted by the unchanged localizer.
    Each full candidate not in narrow is checked once.  A positive count means
    interval pruning would change the original SDF's validity mask.
    """
    t0 = perf_counter()
    narrowed = iter(narrow_chunks)
    remaining = next(narrowed, None)
    full_count = narrow_count = excluded_count = missing_count = 0
    first_missing = []
    for full in full_chunks:
        full = np.asarray(full, dtype=np.int64)
        if not len(full):
            continue
        if np.any(full[1:] <= full[:-1]):
            raise ValueError('Audit requires strictly increasing full candidates')
        full_count += len(full)
        covered = np.zeros(len(full), dtype=np.bool_)
        while remaining is not None:
            remaining = np.asarray(remaining, dtype=np.int64)
            if not len(remaining):
                remaining = next(narrowed, None)
                continue
            if remaining[-1] < full[0]:
                raise ValueError('Narrow candidate was not found in full candidate union')
            if remaining[0] > full[-1]:
                break
            end = int(np.searchsorted(remaining, full[-1], side='right'))
            subset = remaining[:end]
            at = np.searchsorted(full, subset)
            valid = at < len(full)
            if np.any(valid):
                valid[valid] &= full[at[valid]] == subset[valid]
            if not np.all(valid):
                raise ValueError('Narrow candidate was not found in full candidate union')
            covered[at] = True
            narrow_count += len(subset)
            if end == len(remaining):
                remaining = next(narrowed, None)
            else:
                remaining = remaining[end:]
                break
        excluded = full[~covered]
        excluded_count += len(excluded)
        if len(excluded):
            accepted = np.asarray(localize(idx2pts(excluded)), dtype=np.int64)
            if len(accepted):
                if np.any(accepted < 0) or np.any(accepted >= len(excluded)):
                    raise ValueError('Audit localizer returned invalid accepted indices')
                missing_count += len(accepted)
                first_missing.extend(excluded[accepted[:max(0, 8-len(first_missing))]].tolist())
                if stop_on_missing:
                    raise ValueError(
                        '0048 interval candidate pruning UNSAFE: excluded accepted voxels '
                        f'(examples={first_missing[:8]}). Increase interval pad or disable pruning.')
    if remaining is not None or next(narrowed, None) is not None:
        raise ValueError('Narrow candidate was not found in full candidate union')
    return dict(full_candidates=full_count, narrow_candidates=narrow_count,
                excluded_candidates=excluded_count, missing_accepted=missing_count,
                first_missing=first_missing[:8], elapsed_s=perf_counter()-t0)
