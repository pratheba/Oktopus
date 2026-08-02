"""Small, dependency-free I/O helpers for sharing one SDF snug field.

The SDF run is the only place that *builds* the field because it has a signed
avatar distance. UDF/NSDUDF and the direct UDF cutter only load and reuse that
exact field. Nothing happens unless ``shared_snug_field`` is explicitly set in
an adaptation YAML item, so existing no-snug runs stay on their old path.
"""

from __future__ import annotations

import os
import os.path as op
from typing import Dict, Mapping

import numpy as np


def _safe_label(value: object) -> str:
    return str(value).replace("|", "_").replace("/", "_")


def resolve_shared_snug_field_path(
    spec: str,
    *,
    root_path: str,
    output_folder: str,
    item_index: int,
    mode: str,
    target_key: str,
    accessory_key: str,
) -> str:
    """Resolve an absolute or repository-relative field path template."""
    if not spec:
        raise ValueError("shared_snug_field path specification is empty.")

    formatted = str(spec).format(
        index=int(item_index),
        mode=str(mode),
        target=_safe_label(target_key),
        accessory=_safe_label(accessory_key),
        output_folder=op.abspath(output_folder),
    )
    path = op.expanduser(formatted)
    if not op.isabs(path):
        path = op.join(root_path, path)
    return op.abspath(path)


def _validated_field(field: Mapping[str, object]) -> Dict[str, np.ndarray]:
    required = ("scale", "s_bins", "theta_bins")
    missing = [key for key in required if key not in field]
    if missing:
        raise ValueError(f"Shared snug field is missing keys: {missing}")

    result: Dict[str, np.ndarray] = {
        key: np.asarray(value).copy()
        for key, value in field.items()
        if key in {"scale", "delta", "s_bins", "theta_bins", "gap_field", "count"}
    }

    scale = np.asarray(result["scale"], dtype=np.float64)
    s_bins = np.asarray(result["s_bins"], dtype=np.float64).reshape(-1)
    theta_bins = np.asarray(result["theta_bins"], dtype=np.float64).reshape(-1)

    if scale.ndim != 2 or scale.size == 0:
        raise ValueError(
            f"Shared snug scale must be a non-empty 2-D array, got {scale.shape}."
        )
    if s_bins.shape[0] != scale.shape[0]:
        raise ValueError(
            "Shared snug s_bins/scale mismatch: "
            f"{s_bins.shape[0]} vs {scale.shape[0]}."
        )
    if theta_bins.shape[0] != scale.shape[1]:
        raise ValueError(
            "Shared snug theta_bins/scale mismatch: "
            f"{theta_bins.shape[0]} vs {scale.shape[1]}."
        )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(
            "Shared snug scale contains non-finite or non-positive values."
        )
    if not np.all(np.isfinite(s_bins)) or not np.all(np.isfinite(theta_bins)):
        raise ValueError("Shared snug bin coordinates contain non-finite values.")

    result["scale"] = scale
    result["s_bins"] = s_bins
    result["theta_bins"] = theta_bins

    for key in ("delta", "gap_field", "count"):
        if key not in result:
            continue
        value = np.asarray(result[key])
        if value.shape != scale.shape:
            raise ValueError(
                f"Shared snug {key}/scale mismatch: {value.shape} vs {scale.shape}."
            )
        result[key] = value.copy()

    return result


def snug_field_stats(field: Mapping[str, object]) -> str:
    scale = np.asarray(field["scale"], dtype=np.float64)
    return (
        f"shape={tuple(scale.shape)} "
        f"scale_min={float(np.min(scale)):.9g} "
        f"scale_mean={float(np.mean(scale)):.9g} "
        f"scale_max={float(np.max(scale)):.9g}"
    )


def save_shared_snug_field(
    path: str, field: Mapping[str, object]
) -> Dict[str, np.ndarray]:
    """Validate and atomically save the exact SDF-generated field."""
    validated = _validated_field(field)
    parent = op.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    temp_path = path + ".tmp.npz"
    np.savez_compressed(temp_path, **validated)
    os.replace(temp_path, path)
    return validated


def load_shared_snug_field(path: str) -> Dict[str, np.ndarray]:
    """Load a field strictly; opt-in snug must never silently fall back."""
    if not op.isfile(path):
        raise FileNotFoundError(
            "Shared snug field does not exist. Run the SDF adaptation first: "
            f"{path}"
        )
    with np.load(path, allow_pickle=False) as archive:
        field = {key: archive[key] for key in archive.files}
    return _validated_field(field)
