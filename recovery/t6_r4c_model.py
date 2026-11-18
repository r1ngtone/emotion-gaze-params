"""Area-normalized density and optimizer used by the T6 R4c simulation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from fit.t6.pipeline import SCREEN_HEIGHT_PX, SCREEN_WIDTH_CM, SCREEN_WIDTH_PX, VIEWING_DISTANCE_CM


PARAMETER_NAMES = ["Tmax1", "Tmax2", "Tmax3", "Pmax1", "Pmax2", "Pmax3", "dc"]

EMOTION_AREA_COLUMNS = {
    "positive": "positive_area_px2",
    "negative": "negative_area_px2",
    "neutral": "neutral_area_px2",
}

def _raw_attention(time_values: np.ndarray, pmax: float, tmax: float, eps: float) -> np.ndarray:
    values = np.zeros_like(time_values, dtype=np.float64)
    valid = time_values >= 0
    values[valid] = (
        pmax
        * (time_values[valid] / (tmax + eps))
        * np.exp(1.0 - time_values[valid] / (tmax + eps))
    )
    return values

def compute_component_weights_v2(
    t_picture: np.ndarray,
    params: Mapping[str, float],
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    p0_raw = np.ones_like(t_picture, dtype=np.float64)
    p1_raw = _raw_attention(t_picture, float(params["Pmax1"]), float(params["Tmax1"]), eps)
    p2_raw = _raw_attention(t_picture, float(params["Pmax2"]), float(params["Tmax2"]), eps)
    p3_raw = _raw_attention(t_picture, float(params["Pmax3"]), float(params["Tmax3"]), eps)
    total = p0_raw + p1_raw + p2_raw + p3_raw
    if np.any(~np.isfinite(total)) or np.any(total <= 0):
        raise ValueError("T6 v2 组件权重总和必须为正且有限")
    return {
        "central": p0_raw / total,
        "positive": p1_raw / total,
        "negative": p2_raw / total,
        "neutral": p3_raw / total,
    }

def truncated_gaussian_normalizer_v2(
    dc: float,
    screen_width: float = SCREEN_WIDTH_PX,
    screen_height: float = SCREEN_HEIGHT_PX,
) -> float:
    if dc <= 0 or not math.isfinite(dc):
        return math.nan
    x0 = screen_width / 2.0
    y0 = screen_height / 2.0
    x_mass = ndtr((screen_width - x0) / dc) - ndtr(-x0 / dc)
    y_mass = ndtr((screen_height - y0) / dc) - ndtr(-y0 / dc)
    return float(x_mass * y_mass)

def compute_density_v2(
    data: Mapping[str, np.ndarray],
    params: Mapping[str, float],
    eps: float = 1e-12,
) -> np.ndarray:
    values = [float(params[name]) for name in PARAMETER_NAMES]
    if not all(math.isfinite(value) for value in values):
        return np.full(len(data["gaze_x"]), np.nan)
    if min(float(params[name]) for name in ("Tmax1", "Tmax2", "Tmax3")) <= 0:
        return np.full(len(data["gaze_x"]), np.nan)
    if min(float(params[name]) for name in ("Pmax1", "Pmax2", "Pmax3")) < 0:
        return np.full(len(data["gaze_x"]), np.nan)
    dc = float(params["dc"])
    normalizer = truncated_gaussian_normalizer_v2(dc)
    if not math.isfinite(normalizer) or normalizer <= 0:
        return np.full(len(data["gaze_x"]), np.nan)

    x = np.asarray(data["gaze_x"], dtype=np.float64)
    y = np.asarray(data["gaze_y"], dtype=np.float64)
    weights = compute_component_weights_v2(
        np.asarray(data["t_from_picture_start"], dtype=np.float64), params, eps=eps
    )
    x0 = SCREEN_WIDTH_PX / 2.0
    y0 = SCREEN_HEIGHT_PX / 2.0
    gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * dc**2))
    gaussian /= 2.0 * np.pi * dc**2 * normalizer

    density = weights["central"] * gaussian
    for emotion, hit_column in (("positive", "positive"), ("negative", "nagetive"), ("neutral", "neutral")):
        hits = np.asarray(data[hit_column], dtype=np.float64)
        areas = np.asarray(data[EMOTION_AREA_COLUMNS[emotion]], dtype=np.float64)
        valid_area = np.isfinite(areas) & (areas > 0)
        roi_density = np.zeros_like(density)
        roi_density[valid_area] = hits[valid_area] / areas[valid_area]
        density += weights[emotion] * roi_density
    return density

def compute_log_likelihood_v2(
    data: Mapping[str, np.ndarray],
    params: Mapping[str, float],
    eps: float = 1e-300,
) -> float:
    density = compute_density_v2(data, params)
    if np.any(~np.isfinite(density)) or np.any(density < 0):
        return -np.inf
    return float(np.sum(np.log(np.maximum(density, eps))))


REFERENCE_TIME_MS = 1000.0

def temporal_kernel(time_ms: float, tmax_ms: float) -> float:
    if not math.isfinite(time_ms) or time_ms <= 0:
        raise ValueError("time_ms must be positive and finite")
    if not math.isfinite(tmax_ms) or tmax_ms <= 0:
        raise ValueError("tmax_ms must be positive and finite")
    ratio = time_ms / tmax_ms
    return float(ratio * math.exp(1.0 - ratio))

def derive_pmax_preserving_pm(
    tmax: Sequence[float], pm: Sequence[float], target_weight: float, reference_time_ms: float
) -> tuple[float, float, float]:
    """Backsolve one total amplitude while preserving Pmax/sum(Pmax) = pm."""

    tmax_array = np.asarray(tmax, dtype=float)
    pm_array = np.asarray(pm, dtype=float)
    if tmax_array.shape != (3,) or pm_array.shape != (3,):
        raise ValueError("tmax and pm must each contain exactly three values")
    if not np.isfinite(tmax_array).all() or (tmax_array <= 0).any():
        raise ValueError("all Tmax values must be positive and finite")
    if not np.isfinite(pm_array).all() or (pm_array < 0).any():
        raise ValueError("all pm values must be non-negative and finite")
    if not math.isclose(float(pm_array.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("pm values must sum to one")
    if not math.isfinite(target_weight) or not 0.0 < target_weight < 1.0:
        raise ValueError("target_weight must lie strictly between zero and one")

    kernels = np.asarray([temporal_kernel(reference_time_ms, value) for value in tmax_array])
    odds = target_weight / (1.0 - target_weight)
    total_amplitude = odds / float(np.dot(pm_array, kernels))
    return tuple(float(value) for value in total_amplitude * pm_array)

def visual_angle_deg_to_px(degrees: float) -> float:
    if not math.isfinite(degrees) or degrees <= 0:
        raise ValueError("visual angle must be positive and finite")
    size_cm = 2.0 * VIEWING_DISTANCE_CM * math.tan(math.radians(degrees) / 2.0)
    return float(size_cm * SCREEN_WIDTH_PX / SCREEN_WIDTH_CM)


def optimize_log1p(
    log_likelihood: Callable[[Mapping[str, float]], float],
    initial_params: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
    fixed: Mapping[str, float] | None = None,
    maxiter: int = 120,
) -> dict[str, Any]:
    fixed_values = dict(fixed or {})
    unknown = set(initial_params) - set(PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown initial parameters: {sorted(unknown)}")
    free_names = [name for name in PARAMETER_NAMES if name not in fixed_values]
    for name, value in fixed_values.items():
        if name not in PARAMETER_NAMES:
            raise ValueError(f"unknown fixed parameter: {name}")
        lower, upper = bounds[name]
        if not lower <= value <= upper:
            raise ValueError(f"fixed {name}={value} outside [{lower}, {upper}]")

    x0 = np.asarray([math.log1p(float(initial_params[name])) for name in free_names], dtype=float)
    transformed_bounds = [
        (math.log1p(bounds[name][0]), math.log1p(bounds[name][1])) for name in free_names
    ]

    def unpack(theta: np.ndarray) -> dict[str, float]:
        params = {name: float(value) for name, value in fixed_values.items()}
        params.update(
            {name: float(math.expm1(value)) for name, value in zip(free_names, theta)}
        )
        return params

    def objective(theta: np.ndarray) -> float:
        value = float(log_likelihood(unpack(theta)))
        return -value if math.isfinite(value) else 1e100

    started = time.perf_counter()
    result = minimize(
        objective,
        x0=x0,
        bounds=transformed_bounds,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1e-9, "gtol": 1e-5},
    )
    params = unpack(result.x)
    loglike = float(log_likelihood(params))
    valid = math.isfinite(loglike) and all(math.isfinite(value) for value in params.values())
    return {
        "params": params,
        "log_likelihood": loglike,
        "success": bool(result.success and valid),
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
