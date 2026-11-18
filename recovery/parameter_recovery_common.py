#!/usr/bin/env python3
"""Shared deterministic helpers for T60 parameter-recovery preparation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtr


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_CONFIG = SCRIPT_DIR / "recovery_config.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract(config: dict, config_path: Path = DEFAULT_CONFIG) -> dict:
    model_paths = {
        "t5_model": REPO_ROOT / config["t5"]["model_path"],
        "t5_pipeline": REPO_ROOT / config["t5"]["pipeline_path"],
        "t6_model": REPO_ROOT / config["t6"]["model_path"],
    }
    if "t6_v2" in config:
        model_paths["t6_v2_model"] = REPO_ROOT / config["t6_v2"]["model_path"]
    payload = {
        "config": config,
        "config_sha256": sha256_file(config_path),
        "model_sha256": {name: sha256_file(path) for name, path in model_paths.items()},
        "model_paths": {name: str(path) for name, path in model_paths.items()},
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["contract_hash"] = hashlib.sha256(canonical).hexdigest()
    return payload


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % (2**32)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def atomic_write_text(text: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parameter_kind(name: str) -> str:
    if name.endswith("_sigma_s"):
        return "sigma_s"
    if name.endswith("_d_c"):
        return "d_c"
    if name.endswith("_tau"):
        return "tau"
    raise ValueError(f"Unknown T5 parameter: {name}")


def boundary_label(value: float, lower: float, upper: float, fraction: float) -> str:
    tolerance = max((upper - lower) * fraction, 1e-12)
    if value <= lower + tolerance:
        return "lower"
    if value >= upper - tolerance:
        return "upper"
    return "interior"


def robust_maximin_select(
    frame: pd.DataFrame,
    parameter_names: list[str],
    n_select: int,
    id_column: str = "subject_id",
) -> pd.DataFrame:
    if len(frame) < n_select:
        raise ValueError(f"Need {n_select} eligible rows, found {len(frame)}")
    ordered = frame.sort_values(id_column).reset_index(drop=True).copy()
    values = ordered[parameter_names].to_numpy(dtype=float)
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0) * 1.4826
    q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    scale = np.where(mad > 0, mad, np.where(iqr_scale > 0, iqr_scale, 1.0))
    z_values = (values - median) / scale

    center_distance = np.linalg.norm(z_values, axis=1)
    selected = [int(np.argmin(center_distance))]
    remaining = set(range(len(ordered))) - set(selected)
    while len(selected) < n_select:
        best_index = None
        best_distance = -math.inf
        for index in sorted(remaining):
            distance = min(np.linalg.norm(z_values[index] - z_values[chosen]) for chosen in selected)
            if distance > best_distance + 1e-12:
                best_distance = float(distance)
                best_index = index
        if best_index is None:
            raise RuntimeError("Maximin selection failed")
        selected.append(best_index)
        remaining.remove(best_index)

    result = ordered.iloc[selected].copy().reset_index(drop=True)
    result.insert(0, "truth_id", [f"truth_{index:02d}" for index in range(1, n_select + 1)])
    result["selection_order"] = np.arange(1, n_select + 1)
    return result


def gaussian_rectangle_mass(
    x_center: float,
    y_center: float,
    sigma: float,
    width: float,
    height: float,
) -> float:
    if sigma <= 0:
        return math.nan
    x_mass = ndtr((width - x_center) / sigma) - ndtr(-x_center / sigma)
    y_mass = ndtr((height - y_center) / sigma) - ndtr(-y_center / sigma)
    return float(x_mass * y_mass)


def t6_component_weights(time_ms: float, params: dict[str, float], eps: float = 1e-12) -> dict[str, float]:
    raw = [1.0]
    for index in range(1, 4):
        tmax = float(params[f"Tmax{index}"])
        pmax = float(params[f"Pmax{index}"])
        value = 0.0
        if time_ms >= 0:
            value = pmax * (time_ms / (tmax + eps)) * math.exp(1.0 - time_ms / (tmax + eps))
        raw.append(value)
    total = sum(raw) + eps
    return {
        "central": raw[0] / total,
        "positive": raw[1] / total,
        "negative": raw[2] / total,
        "neutral": raw[3] / total,
    }


def t6_screen_mass(
    time_ms: float,
    params: dict[str, float],
    emotion_areas: dict[str, float],
    screen_width: float,
    screen_height: float,
) -> dict[str, float]:
    weights = t6_component_weights(time_ms, params)
    central_mass = gaussian_rectangle_mass(
        screen_width / 2.0,
        screen_height / 2.0,
        float(params["dc"]),
        screen_width,
        screen_height,
    )
    total_mass = weights["central"] * central_mass
    for emotion in ("positive", "negative", "neutral"):
        total_mass += weights[emotion] * float(emotion_areas[emotion])
    return {**weights, "central_screen_mass": central_mass, "total_mass": float(total_mass)}


def t5_initial_nominal_escape_rate(
    d_c: float,
    width: float,
    height: float,
    grid_m: int,
    grid_n: int,
) -> float:
    xs = (np.arange(grid_m) + 0.5) * (width / grid_m)
    ys = (np.arange(grid_n) + 0.5) * (height / grid_n)
    masses = [gaussian_rectangle_mass(float(x), float(y), d_c, width, height) for x in xs for y in ys]
    return float(1.0 - np.mean(masses))


def finite_values(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)
