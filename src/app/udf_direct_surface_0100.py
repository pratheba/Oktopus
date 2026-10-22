"""0100: direct, three-state learned-UDF consistency on a fused SDF surface.

Unlike 0079/0081 this does not use connected components or propagate seeds.
Only VALID model evaluations can establish an absent/present surface.  An
unsupported UDF query is UNKNOWN, never a manufactured high-distance value.
"""
from __future__ import annotations
import numpy as np


def classify_consistency(assigned, valid, supported, absent):
    """Combine all assigned model owners; valid support from any owner vetoes a cut.

    `supported` is the count of valid owners with distance <= grow threshold.
    `absent` is the count of valid owners with distance >= seed threshold.
    Ambiguous scores between those thresholds are explicitly unknown.
    """
    a, v, s, n = (np.asarray(x, dtype=np.int32).reshape(-1) for x in
                  (assigned, valid, supported, absent))
    if not (len(a) == len(v) == len(s) == len(n)):
        raise ValueError('0100 mismatched face arrays')
    if np.any((a < 0) | (v < 0) | (s < 0) | (n < 0) | (v > a) | ((s + n) > v)):
        raise ValueError('0100 inconsistent direct owner evidence')
    keep = s > 0
    remove = (a > 0) & (v == a) & (n == a) & ~keep
    unknown = ~(keep | remove)
    return keep, remove, unknown


def classify_owner_scores(values, valid, *, grow, seed, far):
    score = np.asarray(values, dtype=np.float64)
    validity = np.asarray(valid, bool) & np.isfinite(score)
    if not (0 <= float(grow) < float(seed) < float(far)):
        raise ValueError('0100 requires 0 <= grow < seed < far; do not use saturated far sentinel')
    return validity, validity & (score <= grow), validity & (score >= seed) & (score < far - 1e-6)
