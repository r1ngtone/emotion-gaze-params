#!/usr/bin/env python3
"""Run the frozen, resumable T6 normalized-observation parameter recovery."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_t6_normalized_smoke as observation  # noqa: E402
from parameter_recovery_common import atomic_write_csv, atomic_write_text, stable_seed  # noqa: E402


TASK_ID = "T20260726-003-t6-formal-recovery-run"
DESIGN_DIR = SCRIPT_DIR / "outputs" / "T6_v1" / "formal_recovery_design_v2_field_common"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "T6_v1" / "formal_recovery_v2_field_common"
PARAMETERS = ("Tmax1", "Tmax2", "Tmax3", "pm_positive", "pm_negative", "pm_neutral", "dc")
EMOTIONS = ("positive", "negative", "neutral")
SCREEN_WIDTH = 1920.0
SCREEN_HEIGHT = 1080.0
SCREEN_WIDTH_CM = 34.5
VIEWING_DISTANCE_CM = 57.0
EPS = 1e-15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DESIGN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max-datasets", type=int, default=None, help="Run only the first N datasets; test mode only.")
    parser.add_argument("--template-stride", type=int, default=1, help="Use every Nth actual timestamp; values >1 are test only.")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def validate_design(design_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contract_path = design_dir / "formal-design-contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    recorded_hash = contract.pop("contract_hash")
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    computed_hash = hashlib.sha256(canonical).hexdigest()
    contract["contract_hash"] = recorded_hash
    if recorded_hash != computed_hash:
        raise ValueError("Frozen design contract hash mismatch")
    if contract.get("status") != "DESIGN_V2_FIELD_COMMON_READY_FOR_RUN":
        raise ValueError("Frozen design is not ready for the formal run")
    if contract.get("formal_outcomes_generated") is not False:
        raise ValueError("Frozen design unexpectedly claims formal outcomes")
    truths = pd.read_csv(design_dir / "formal-truth-bank.csv", dtype={"subject_id": str})
    starts = pd.read_csv(design_dir / "blind-start-bank.csv")
    empirical = pd.read_csv(design_dir / "empirical-parameter-summary.csv")
    if len(truths) != int(contract["truth_count"]) or truths["truth_id"].nunique() != len(truths):
        raise ValueError("Truth bank does not match the frozen contract")
    if len(starts) != int(contract["blind_starts"]) or starts["start_id"].nunique() != len(starts):
        raise ValueError("Blind-start bank does not match the frozen contract")
    if not truths["trial_count"].eq(int(contract["trials_per_dataset"])).all():
        raise ValueError("Truth templates do not match the frozen trial count")
    return contract, truths, starts, empirical


def normalize_emotion(value: Any) -> str:
    text = str(value).strip().lower()
    return {"postive": "positive", "positve": "positive", "nagetive": "negative"}.get(text, text)


def parse_pair(value: Any) -> tuple[float, float]:
    pair = ast.literal_eval(str(value))
    return float(pair[0]), float(pair[1])


def degrees_to_px(degrees: float) -> float:
    cm = 2.0 * VIEWING_DISTANCE_CM * math.tan(math.radians(float(degrees)) / 2.0)
    return cm * SCREEN_WIDTH / SCREEN_WIDTH_CM


def px_to_degrees(px: float) -> float:
    cm = float(px) * SCREEN_WIDTH_CM / SCREEN_WIDTH
    return math.degrees(2.0 * math.atan(cm / (2.0 * VIEWING_DISTANCE_CM)))


def trial_bounds(records: pd.DataFrame, trial_index: int) -> np.ndarray:
    row = records.iloc[trial_index]
    side_specs = (
        ("左情绪类型", "左图片坐标", "左图片大小"),
        ("中情绪类型", "中图片坐标", "中图片大小"),
        ("右情绪类型", "右图片坐标", "右图片大小"),
    )
    mapping: dict[str, tuple[float, float, float, float]] = {}
    for emotion_column, position_column, size_column in side_specs:
        emotion = normalize_emotion(row[emotion_column])
        x, y = parse_pair(row[position_column])
        width, height = parse_pair(row[size_column])
        mapping[emotion] = (x, x + width, y, y + height)
    if set(mapping) != set(EMOTIONS):
        raise ValueError(f"Trial {trial_index + 1} does not contain exactly the three emotions")
    return np.asarray([mapping[emotion] for emotion in EMOTIONS], dtype=float)


def first_trigger_index(frame: pd.DataFrame, trigger: int) -> int:
    indices = frame.index[frame["trigger"].eq(trigger)].tolist()
    if not indices:
        raise ValueError(f"Missing trigger {trigger}")
    return int(indices[0])


def load_actual_template(truth: Mapping[str, Any], stride: int = 1) -> dict[str, np.ndarray]:
    if stride < 1:
        raise ValueError("template stride must be positive")
    eyetrack = pd.read_csv(
        Path(str(truth["eyetrack_path"])),
        usecols=["timestamp", "bino_eye_valid", "trigger"],
    )
    records = pd.read_csv(Path(str(truth["records_path"])))
    for column in ("timestamp", "bino_eye_valid", "trigger"):
        eyetrack[column] = pd.to_numeric(eyetrack[column], errors="coerce")
    if len(records) != 40:
        raise ValueError(f"Expected 40 records, found {len(records)}")
    times: list[np.ndarray] = []
    trials: list[np.ndarray] = []
    bounds: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    for trial in range(1, 41):
        fix_index = first_trigger_index(eyetrack, trial * 100)
        picture_index = first_trigger_index(eyetrack, trial * 100 + 1)
        blank_index = first_trigger_index(eyetrack, trial * 100 + 2)
        if not (fix_index < picture_index < blank_index):
            raise ValueError(f"Non-monotonic trigger indices for trial {trial}")
        selected = eyetrack.loc[fix_index:blank_index - 1].copy()
        selected = selected.loc[selected["bino_eye_valid"].eq(1)].iloc[::stride]
        if selected.empty:
            continue
        picture_timestamp = float(eyetrack.loc[picture_index, "timestamp"])
        time_values = (selected["timestamp"].to_numpy(dtype=float) - picture_timestamp) / 1e6
        phase_values = (selected.index.to_numpy() >= picture_index).astype(int)
        rectangle = trial_bounds(records, trial - 1)
        times.append(time_values)
        trials.append(np.full(len(selected), trial, dtype=int))
        bounds.append(np.repeat(rectangle[None, :, :], len(selected), axis=0))
        phases.append(phase_values)
    template = {
        "trial_number": np.concatenate(trials),
        "t_picture_ms": np.concatenate(times),
        "roi_bounds": np.concatenate(bounds),
        "picture": np.concatenate(phases),
    }
    template["scheduled_trials"] = np.asarray([40], dtype=int)
    template["observed_trials"] = np.asarray([np.unique(template["trial_number"]).size], dtype=int)
    if not (np.any(template["t_picture_ms"] < 0) and np.any(template["t_picture_ms"] >= 0)):
        raise ValueError("Actual template must contain fixation and picture timestamps")
    return template


def external_to_internal(params: Mapping[str, float]) -> dict[str, float]:
    return {
        "Tmax1": float(params["Tmax1"]),
        "Tmax2": float(params["Tmax2"]),
        "Tmax3": float(params["Tmax3"]),
        "pm_positive": float(params["pm_positive"]),
        "pm_negative": float(params["pm_negative"]),
        "pm_neutral": float(params["pm_neutral"]),
        "dc": degrees_to_px(float(params["dc"])),
    }


def encode(params: Mapping[str, float]) -> np.ndarray:
    pm = np.asarray([params[f"pm_{emotion}"] for emotion in EMOTIONS], dtype=float)
    logits = observation.pm_to_logits(pm)
    return np.asarray([params["Tmax1"], params["Tmax2"], params["Tmax3"], *logits, params["dc"]], dtype=float)


def decode(theta: np.ndarray) -> dict[str, float]:
    values = np.asarray(theta, dtype=float)
    pm = observation.logits_to_pm(values[3:5])
    return {
        "Tmax1": float(values[0]), "Tmax2": float(values[1]), "Tmax3": float(values[2]),
        "pm_positive": float(pm[0]), "pm_negative": float(pm[1]), "pm_neutral": float(pm[2]),
        "dc": float(values[5]),
    }


def vector_screen_gaussian(rng: np.random.Generator, count: int, dc_px: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.empty(count, dtype=float)
    y = np.empty(count, dtype=float)
    pending = np.arange(count)
    for _ in range(1000):
        if len(pending) == 0:
            return x, y
        candidate_x = rng.normal(observation.CENTER_X, dc_px, size=len(pending))
        candidate_y = rng.normal(observation.CENTER_Y, dc_px, size=len(pending))
        valid = ((candidate_x >= 0) & (candidate_x <= SCREEN_WIDTH)
                 & (candidate_y >= 0) & (candidate_y <= SCREEN_HEIGHT))
        accepted = pending[valid]
        x[accepted] = candidate_x[valid]
        y[accepted] = candidate_y[valid]
        pending = pending[~valid]
    raise RuntimeError("Failed to sample the truncated screen Gaussian")


def generate_dataset(template: Mapping[str, np.ndarray], truth: Mapping[str, float], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    internal = external_to_internal(truth)
    times = np.asarray(template["t_picture_ms"], dtype=float)
    roi_bounds = np.asarray(template["roi_bounds"], dtype=float)
    weights = observation.mixture_weights(times, internal)
    uniform = rng.random(len(times))
    components = (uniform[:, None] > np.cumsum(weights, axis=1)).sum(axis=1)
    x = np.empty(len(times), dtype=float)
    y = np.empty(len(times), dtype=float)
    central = components == 0
    x[central], y[central] = vector_screen_gaussian(rng, int(central.sum()), internal["dc"])
    for component in range(1, 4):
        selected = components == component
        rectangles = roi_bounds[selected, component - 1, :]
        x[selected] = rng.uniform(rectangles[:, 0], rectangles[:, 1])
        y[selected] = rng.uniform(rectangles[:, 2], rectangles[:, 3])
    return {
        "trial_number": np.asarray(template["trial_number"], dtype=int),
        "t_picture_ms": times,
        "gaze_x": x,
        "gaze_y": y,
        "roi_bounds": roi_bounds,
    }


def data_digest(data: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("trial_number", "t_picture_ms", "gaze_x", "gaze_y", "roi_bounds"):
        digest.update(np.ascontiguousarray(data[key]).tobytes())
    return digest.hexdigest()


def log_likelihood(data: Mapping[str, np.ndarray], external: Mapping[str, float]) -> float:
    density = observation.normalized_density(data, external_to_internal(external))
    return float(np.log(density).sum())


def prepare_likelihood(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    x = np.asarray(data["gaze_x"], dtype=float)
    y = np.asarray(data["gaze_y"], dtype=float)
    bounds = np.asarray(data["roi_bounds"], dtype=float)
    hit_density = np.empty((len(x), 3), dtype=float)
    for index in range(3):
        left, right, top, bottom = (bounds[:, index, column] for column in range(4))
        hit = (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
        hit_density[:, index] = hit.astype(float) / ((right - left) * (bottom - top))
    return {
        "time": np.asarray(data["t_picture_ms"], dtype=float),
        "radial_squared": (x - observation.CENTER_X) ** 2 + (y - observation.CENTER_Y) ** 2,
        "hit_density": hit_density,
    }


def likelihood_value_and_gradient(theta: np.ndarray, prepared: Mapping[str, np.ndarray]) -> tuple[float, np.ndarray]:
    values = np.asarray(theta, dtype=float)
    external = decode(values)
    time_values = np.asarray(prepared["time"], dtype=float)
    radial_squared = np.asarray(prepared["radial_squared"], dtype=float)
    hit_density = np.asarray(prepared["hit_density"], dtype=float)
    pm = np.asarray([external[f"pm_{emotion}"] for emotion in EMOTIONS], dtype=float)
    tmax = values[:3]
    valid = time_values >= 0
    shape = np.zeros((len(time_values), 3), dtype=float)
    for index in range(3):
        scaled = time_values[valid] / tmax[index]
        shape[valid, index] = scaled * np.exp(1.0 - scaled)
    raw = observation.ACTIVATION_SCALE * shape * pm[None, :]
    denominator = 1.0 + raw.sum(axis=1)

    degrees = float(values[5])
    radians_half = math.radians(degrees) / 2.0
    conversion = 2.0 * VIEWING_DISTANCE_CM * SCREEN_WIDTH / SCREEN_WIDTH_CM
    sigma = conversion * math.tan(radians_half)
    dsigma_ddegree = conversion * (math.pi / 360.0) / (math.cos(radians_half) ** 2)
    ax = observation.CENTER_X / sigma
    ay = observation.CENTER_Y / sigma
    mass_x = 2.0 * observation.ndtr(ax) - 1.0
    mass_y = 2.0 * observation.ndtr(ay) - 1.0
    normalizer_mass = mass_x * mass_y
    gaussian = np.exp(-radial_squared / (2.0 * sigma**2)) / (2.0 * np.pi * sigma**2 * normalizer_mass)
    numerator = gaussian + (raw * hit_density).sum(axis=1)
    numerator = np.clip(numerator, EPS, None)
    value = float(np.sum(np.log(numerator) - np.log(denominator)))

    gradient = np.zeros(6, dtype=float)
    for index in range(3):
        draw = raw[:, index] * (time_values - tmax[index]) / (tmax[index] ** 2)
        draw[~valid] = 0.0
        gradient[index] = np.sum(draw * hit_density[:, index] / numerator - draw / denominator)
    for logit_index in range(2):
        dpm = pm * (-pm[logit_index])
        dpm[logit_index] += pm[logit_index]
        draw = observation.ACTIVATION_SCALE * shape * dpm[None, :]
        gradient[3 + logit_index] = np.sum(
            (draw * hit_density).sum(axis=1) / numerator - draw.sum(axis=1) / denominator
        )
    phi_x = math.exp(-0.5 * ax**2) / math.sqrt(2.0 * math.pi)
    phi_y = math.exp(-0.5 * ay**2) / math.sqrt(2.0 * math.pi)
    dlogmass_dsigma = (-2.0 * ax * phi_x / sigma) / mass_x + (-2.0 * ay * phi_y / sigma) / mass_y
    dloggaussian_dsigma = radial_squared / sigma**3 - 2.0 / sigma - dlogmass_dsigma
    dgaussian_ddegree = gaussian * dloggaussian_dsigma * dsigma_ddegree
    gradient[5] = np.sum(dgaussian_ddegree / numerator)
    return value, gradient


def fit_dataset(
    data: Mapping[str, np.ndarray], start: Mapping[str, float],
    fit_bounds: Mapping[str, list[float]], maxiter: int,
) -> dict[str, Any]:
    x0 = encode(start)
    bounds = [tuple(fit_bounds[f"Tmax{i}"]) for i in range(1, 4)] + [
        tuple(fit_bounds["pm_logit_positive"]), tuple(fit_bounds["pm_logit_negative"]),
        tuple(fit_bounds["dc_degree"]),
    ]
    started = time.perf_counter()
    prepared = prepare_likelihood(data)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = likelihood_value_and_gradient(theta, prepared)
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            return 1e100, np.zeros_like(theta)
        return -value, -gradient

    result = minimize(
        objective, x0=x0, jac=True, bounds=bounds, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 30},
    )
    fitted = decode(result.x)
    return {
        "success": bool(result.success), "message": str(result.message),
        "log_likelihood": float(-result.fun), "runtime_seconds": float(time.perf_counter() - started),
        "optimizer_iterations": int(result.nit), "optimizer_evaluations": int(result.nfev),
        "fitted_logit_positive": float(result.x[3]), "fitted_logit_negative": float(result.x[4]),
        **fitted,
    }


def checkpoint_path(checkpoint_dir: Path, truth_id: str, repeat: int) -> Path:
    return checkpoint_dir / f"{truth_id}__R{repeat:03d}.json"


def run_one_dataset(
    truth: Mapping[str, Any], starts: list[dict[str, Any]], template: Mapping[str, np.ndarray],
    repeat: int, base_seed: int, design_hash: str, fit_bounds: Mapping[str, list[float]], maxiter: int,
) -> dict[str, Any]:
    truth_id = str(truth["truth_id"])
    truth_params = {name: float(truth[name]) for name in PARAMETERS}
    seed = stable_seed(base_seed, design_hash, truth_id, repeat)
    data = generate_dataset(template, truth_params, seed)
    repeated = generate_dataset(template, truth_params, seed)
    reproducible = data_digest(data) == data_digest(repeated)
    truth_ll = log_likelihood(data, truth_params)
    fit_rows = []
    for start in starts:
        start_params = {name: float(start[name]) for name in PARAMETERS}
        fit = fit_dataset(data, start_params, fit_bounds, maxiter)
        fit_rows.append({
            "dataset_id": f"{truth_id}-R{repeat:03d}", "truth_id": truth_id,
            "repeat": repeat, "seed": seed, "start_id": str(start["start_id"]),
            **{f"truth_{name}": truth_params[name] for name in PARAMETERS},
            **{f"fitted_{name}": fit[name] for name in PARAMETERS},
            **{key: fit[key] for key in (
                "success", "message", "log_likelihood", "runtime_seconds",
                "optimizer_iterations", "optimizer_evaluations",
                "fitted_logit_positive", "fitted_logit_negative",
            )},
        })
    valid = [row for row in fit_rows if row["success"] and math.isfinite(row["log_likelihood"])]
    candidates = valid or [row for row in fit_rows if math.isfinite(row["log_likelihood"])]
    if not candidates:
        raise RuntimeError(f"No finite fit for {truth_id} repeat {repeat}")
    best = max(candidates, key=lambda row: row["log_likelihood"])
    manifest = {
        "dataset_id": best["dataset_id"], "truth_id": truth_id, "repeat": repeat, "seed": seed,
        "scheduled_trials": int(template["scheduled_trials"][0]),
        "observed_trials": int(template["observed_trials"][0]), "n_observations": len(data["gaze_x"]),
        "seed_reproducible": reproducible,
        "finite_density": bool(np.isfinite(observation.normalized_density(data, external_to_internal(truth_params))).all()),
        "truth_log_likelihood": truth_ll, "best_log_likelihood": best["log_likelihood"],
        "best_not_below_truth": bool(best["log_likelihood"] >= truth_ll - 1e-6),
        "successful_fits": len(valid), "expected_fits": len(starts),
    }
    return {"manifest": manifest, "fits": fit_rows, "best": best}


def run_truth_batch(payload: Mapping[str, Any]) -> dict[str, Any]:
    truth = payload["truth"]
    starts = payload["starts"]
    repeats = payload["repeats"]
    checkpoint_dir = Path(payload["checkpoint_dir"])
    template = load_actual_template(truth, int(payload["template_stride"]))
    completed = 0
    for repeat in repeats:
        result = run_one_dataset(
            truth, starts, template, int(repeat), int(payload["base_seed"]), str(payload["design_hash"]),
            payload["fit_bounds"], int(payload["maxiter"]),
        )
        atomic_json(result, checkpoint_path(checkpoint_dir, str(truth["truth_id"]), int(repeat)))
        completed += 1
        print(
            f"[T6 formal] {truth['truth_id']} R{int(repeat):03d} "
            f"fits={result['manifest']['successful_fits']}/{result['manifest']['expected_fits']}",
            flush=True,
        )
    return {"truth_id": str(truth["truth_id"]), "completed": completed, "observations": len(template["t_picture_ms"])}


def load_checkpoints(checkpoint_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifests: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    best: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("truth_*__R*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(payload["manifest"])
        fits.extend(payload["fits"])
        best.append(payload["best"])
    return pd.DataFrame(manifests), pd.DataFrame(fits), pd.DataFrame(best)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    result = spearmanr(x, y)
    return float(getattr(result, "statistic", getattr(result, "correlation", result[0])))


def summarize_results(
    best: pd.DataFrame, fits: pd.DataFrame, manifest: pd.DataFrame, truths: pd.DataFrame,
    empirical: pd.DataFrame, bootstrap_repeats: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    empirical_spans = empirical.set_index("parameter")["p05_p95_span"].to_dict()
    long_rows = []
    for row in best.to_dict("records"):
        for parameter in PARAMETERS:
            error = float(row[f"fitted_{parameter}"]) - float(row[f"truth_{parameter}"])
            long_rows.append({
                "dataset_id": row["dataset_id"], "truth_id": row["truth_id"], "repeat": row["repeat"],
                "parameter": parameter, "truth": float(row[f"truth_{parameter}"]),
                "recovered": float(row[f"fitted_{parameter}"]), "error": error,
                "absolute_error": abs(error), "normalized_error": error / float(empirical_spans[parameter]),
            })
    long = pd.DataFrame(long_rows)
    per_truth_rows = []
    for (truth_id, parameter), group in long.groupby(["truth_id", "parameter"], sort=True):
        error = group["error"].to_numpy(dtype=float)
        per_truth_rows.append({
            "truth_id": truth_id, "parameter": parameter, "truth": float(group["truth"].iloc[0]),
            "n_repeats": len(group), "median_recovered": float(group["recovered"].median()),
            "mean_recovered": float(group["recovered"].mean()), "bias": float(error.mean()),
            "mcse_bias": float(error.std(ddof=1) / math.sqrt(len(error))),
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "median_absolute_error": float(np.median(np.abs(error))),
            "normalized_bias": float(group["normalized_error"].mean()),
            "normalized_mae": float(group["normalized_error"].abs().mean()),
        })
    per_truth = pd.DataFrame(per_truth_rows)
    rng = np.random.default_rng(seed)
    summary_rows = []
    fit_bounds = {
        "Tmax1": (1.0, 3500.0), "Tmax2": (1.0, 3500.0), "Tmax3": (1.0, 3500.0),
        "dc": (0.1, 10.0),
    }
    for parameter in PARAMETERS:
        detail = long.loc[long["parameter"].eq(parameter)]
        grouped = per_truth.loc[per_truth["parameter"].eq(parameter)].sort_values("truth_id")
        truth_values = grouped["truth"].to_numpy(dtype=float)
        rho = safe_spearman(truth_values, grouped["median_recovered"].to_numpy(dtype=float))
        samples = {truth_id: detail.loc[detail["truth_id"].eq(truth_id), "recovered"].to_numpy(dtype=float)
                   for truth_id in grouped["truth_id"]}
        boot = np.empty(bootstrap_repeats, dtype=float)
        boot_bias = np.empty(bootstrap_repeats, dtype=float)
        boot_mae = np.empty(bootstrap_repeats, dtype=float)
        boot_rmse = np.empty(bootstrap_repeats, dtype=float)
        truth_by_id = dict(zip(grouped["truth_id"], truth_values))
        for index in range(bootstrap_repeats):
            selected = {
                truth_id: rng.choice(samples[truth_id], len(samples[truth_id]), replace=True)
                for truth_id in grouped["truth_id"]
            }
            medians = [np.median(selected[truth_id]) for truth_id in grouped["truth_id"]]
            boot[index] = safe_spearman(truth_values, np.asarray(medians))
            sampled_errors = np.concatenate([
                selected[truth_id] - truth_by_id[truth_id] for truth_id in grouped["truth_id"]
            ])
            boot_bias[index] = sampled_errors.mean()
            boot_mae[index] = np.abs(sampled_errors).mean()
            boot_rmse[index] = np.sqrt(np.mean(sampled_errors ** 2))
        error = detail["error"].to_numpy(dtype=float)
        normalized = detail["normalized_error"].to_numpy(dtype=float)
        if parameter.startswith("pm_"):
            logit_name = "fitted_logit_positive" if parameter.endswith("positive") else (
                "fitted_logit_negative" if parameter.endswith("negative") else None)
            if logit_name is None:
                boundary_fraction = float(((best["fitted_pm_neutral"] <= observation.logits_to_pm(np.array([6.0, 6.0]))[2] + 1e-12)
                                           | (best["fitted_pm_neutral"] >= observation.logits_to_pm(np.array([-6.0, -6.0]))[2] - 1e-12)).mean())
            else:
                boundary_fraction = float(best[logit_name].abs().ge(6.0 - 0.012).mean())
        else:
            lower, upper = fit_bounds[parameter]
            tolerance = 0.001 * (upper - lower)
            boundary_fraction = float((detail["recovered"].le(lower + tolerance)
                                       | detail["recovered"].ge(upper - tolerance)).mean())
        summary_rows.append({
            "parameter": parameter, "unit": "ms" if parameter.startswith("Tmax") else ("degree" if parameter == "dc" else "proportion"),
            "n_truths": grouped["truth_id"].nunique(), "repeats_per_truth": int(grouped["n_repeats"].min()),
            "n_datasets": len(detail), "truth_level_spearman_rho": rho,
            "rho_bootstrap_95pct_low": float(np.nanquantile(boot, 0.025)),
            "rho_bootstrap_95pct_high": float(np.nanquantile(boot, 0.975)),
            "bias": float(error.mean()), "mcse_bias": float(error.std(ddof=1) / math.sqrt(len(error))),
            "bias_bootstrap_95pct_low": float(np.quantile(boot_bias, 0.025)),
            "bias_bootstrap_95pct_high": float(np.quantile(boot_bias, 0.975)),
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae_bootstrap_95pct_low": float(np.quantile(boot_mae, 0.025)),
            "mae_bootstrap_95pct_high": float(np.quantile(boot_mae, 0.975)),
            "rmse_bootstrap_95pct_low": float(np.quantile(boot_rmse, 0.025)),
            "rmse_bootstrap_95pct_high": float(np.quantile(boot_rmse, 0.975)),
            "median_absolute_error": float(np.median(np.abs(error))),
            "normalized_bias": float(normalized.mean()), "normalized_mae": float(np.abs(normalized).mean()),
            "normalized_rmse": float(np.sqrt(np.mean(normalized ** 2))), "boundary_fraction": boundary_fraction,
        })
    summary = pd.DataFrame(summary_rows)
    truth_wide = truths.set_index("truth_id")[list(PARAMETERS)].astype(float)
    recovered_wide = per_truth.pivot(index="truth_id", columns="parameter", values="median_recovered")[list(PARAMETERS)]
    error_wide = long.pivot(index="dataset_id", columns="parameter", values="error")[list(PARAMETERS)]
    cross = pd.DataFrame(index=PARAMETERS, columns=PARAMETERS, dtype=float)
    for truth_parameter in PARAMETERS:
        for recovered_parameter in PARAMETERS:
            cross.loc[truth_parameter, recovered_parameter] = safe_spearman(
                truth_wide[truth_parameter].to_numpy(), recovered_wide[recovered_parameter].to_numpy(),
            )
    matrices = {
        "truth-correlation-matrix": truth_wide.corr(method="spearman"),
        "recovered-median-correlation-matrix": recovered_wide.corr(method="spearman"),
        "truth-to-recovered-cross-correlation-matrix": cross,
        "within-truth-error-correlation-matrix": error_wide.corr(method="spearman"),
    }
    implementation = pd.DataFrame([{
        "datasets": best["dataset_id"].nunique(), "truths": best["truth_id"].nunique(),
        "repeats_per_truth_min": int(best.groupby("truth_id").size().min()),
        "repeats_per_truth_max": int(best.groupby("truth_id").size().max()),
        "all_fit_success_fraction": float(fits["success"].astype(bool).mean()),
        "selected_fit_success_fraction": float(best["success"].astype(bool).mean()),
        "finite_estimate_fraction": float(np.isfinite(best[[f"fitted_{p}" for p in PARAMETERS]]).all(axis=1).mean()),
        "seed_reproducible_fraction": float(manifest["seed_reproducible"].astype(bool).mean()),
        "finite_density_fraction": float(manifest["finite_density"].astype(bool).mean()),
        "best_not_below_truth_fraction": float(manifest["best_not_below_truth"].astype(bool).mean()),
        "likelihood_gap_median": float((manifest["best_log_likelihood"] - manifest["truth_log_likelihood"]).median()),
        "likelihood_gap_p05": float((manifest["best_log_likelihood"] - manifest["truth_log_likelihood"]).quantile(0.05)),
        "likelihood_gap_min": float((manifest["best_log_likelihood"] - manifest["truth_log_likelihood"]).min()),
        "bootstrap_repeats": bootstrap_repeats,
    }])
    return summary, per_truth, matrices, implementation


def make_figures(
    per_truth: pd.DataFrame, summary: pd.DataFrame,
    matrices: Mapping[str, pd.DataFrame], output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 7.0))
    for ax, parameter in zip(axes.ravel(), PARAMETERS):
        frame = per_truth.loc[per_truth["parameter"].eq(parameter)].sort_values("truth")
        ax.scatter(frame["truth"], frame["median_recovered"], color="#4C78A8", s=24)
        limits = [min(frame["truth"].min(), frame["median_recovered"].min()),
                  max(frame["truth"].max(), frame["median_recovered"].max())]
        ax.plot(limits, limits, "--", color="black", linewidth=0.8)
        ax.set_title(parameter)
        ax.set_xlabel("True value")
        ax.set_ylabel("Median recovered value")
    axes.ravel()[-1].axis("off")
    fig.suptitle("T6 formal truth-level parameter recovery")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"figure-01-truth-vs-recovered.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(summary["parameter"], summary["normalized_mae"], color="#4C78A8")
    ax.set_ylabel("MAE / empirical P05-P95 span")
    ax.set_title("T6 normalized parameter-recovery error")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"figure-02-normalized-error.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)

    selected = (
        ("truth-correlation-matrix", "True parameters"),
        ("recovered-median-correlation-matrix", "Recovered medians"),
        ("within-truth-error-correlation-matrix", "Within-truth recovery errors"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    image = None
    for ax, (key, title) in zip(axes, selected):
        matrix = matrices[key].loc[list(PARAMETERS), list(PARAMETERS)].to_numpy(float)
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(np.arange(len(PARAMETERS)), PARAMETERS, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(PARAMETERS)), PARAMETERS, fontsize=7)
        ax.set_title(title)
    colorbar_axis = fig.add_axes([0.95, 0.25, 0.012, 0.52])
    fig.colorbar(image, cax=colorbar_axis, label="Spearman correlation")
    fig.suptitle("T6 parameter dependence and recovery-error confusion")
    fig.subplots_adjust(left=0.06, right=0.925, bottom=0.22, top=0.84, wspace=0.30)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"figure-03-parameter-confusion.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def render_reports(
    summary: pd.DataFrame, implementation: pd.DataFrame, manifest: pd.DataFrame,
    matrices: Mapping[str, pd.DataFrame], optimizer_by_truth: pd.DataFrame,
    pm_composition_summary: pd.DataFrame,
    run_contract: Mapping[str, Any], output_dir: Path,
) -> None:
    error_corr = matrices["within-truth-error-correlation-matrix"].copy()
    np.fill_diagonal(error_corr.values, np.nan)
    location = np.unravel_index(np.nanargmax(np.abs(error_corr.values)), error_corr.shape)
    error_pair = (error_corr.index[location[0]], error_corr.columns[location[1]])
    error_value = float(error_corr.iloc[location])
    weakest_optimizer = optimizer_by_truth.sort_values("best_not_below_truth_fraction").iloc[0]
    report = f"""# T6正式参数恢复分析

## 分析问题

在当前40-trial快速三图任务、经验真值范围和实际时间网格/图片布局下，Tmax、pm和dc能否由同一规范化观测模型稳定恢复。

## 正式运行

- 真值：{int(implementation.loc[0, 'truths'])}组。
- 每组独立重复：{int(implementation.loc[0, 'repeats_per_truth_min'])}次。
- 模拟数据集：{int(implementation.loc[0, 'datasets'])}个。
- 全部盲起点拟合成功比例：{implementation.loc[0, 'all_fit_success_fraction']:.4f}。
- 最佳拟合成功比例：{implementation.loc[0, 'selected_fit_success_fraction']:.4f}。
- 有限参数估计比例：{implementation.loc[0, 'finite_estimate_fraction']:.4f}。
- 固定随机种子复现比例：{manifest['seed_reproducible'].mean():.4f}。
- 最佳盲起点似然达到或超过生成真值似然的比例：{implementation.loc[0, 'best_not_below_truth_fraction']:.4f}。

## 连续恢复证据

```csv
{summary.to_csv(index=False).strip()}
```

## 参数混淆

真值组内恢复误差的最大绝对参数间相关为{abs(error_value):.3f}（{error_pair[0]}与{error_pair[1]}，方向{error_value:.3f}）。完整真值、恢复中位数、真值—恢复值和误差相关矩阵均随结果保存。

`pm`组成距离：Aitchison距离中位数为{pm_composition_summary.loc[0, 'aitchison_distance_median']:.4f}，95%分位数为{pm_composition_summary.loc[0, 'aitchison_distance_p95']:.4f}；总变差距离中位数为{pm_composition_summary.loc[0, 'total_variation_distance_median']:.4f}。

## 优化器覆盖

全部拟合均返回成功且估计有限，但盲起点并非在每个数据集都达到生成真值的似然。最低的真值配置为{weakest_optimizer['truth_id']}，其比例为{weakest_optimizer['best_not_below_truth_fraction']:.3f}，似然差中位数为{weakest_optimizer['likelihood_gap_median']:.3f}。这表明形式收敛不等于已找到更优似然区域；逐真值结果和优化器审计必须与总体相关共同解释。

## 结论边界

本报告不使用统一二元通过线。每个参数族依据真值—恢复值关系、误差及Monte Carlo不确定性、边界率、失败率、参数空间局部表现和参数混淆共同解释。正式结果需要作者复核后才能进入论文。
"""
    appendix = f"""# T6正式参数恢复统计附录

## 统计单位

- 排序恢复：每组真值重复估计的中位数，再跨真值计算Spearman相关。
- 误差：模拟数据集层级，同时按真值分层。
- 相关区间：每组真值内分层bootstrap，重复{int(implementation.loc[0, 'bootstrap_repeats'])}次。
- 偏差Monte Carlo标准误：误差标准差除以模拟次数平方根。
- pm三个分量构成闭合组成，不按三个独立自由变量解释。

## 实现检查

```csv
{implementation.to_csv(index=False).strip()}
```
"""
    catalog = """# Figure catalog

## Figure 1: truth versus recovered

- 目的：检查各参数在经验真值范围内是否维持真值—恢复值关系。
- 依据：每组真值正式重复的恢复中位数。
- 观察：结合参数汇总表中的相关区间和误差判断。
- 限制：跨参数单位不同，不比较点云绝对跨度。

## Figure 2: normalized error

- 目的：以当前N118经验P05-P95跨度为分母比较参数恢复误差。
- 依据：所有正式模拟数据集的MAE。
- 观察：柱高表示相对经验范围的误差大小。
- 限制：标准化仅服务于跨参数展示，不是通过阈值。

## Figure 3: parameter confusion

- 目的：区分真值库相关、恢复中位数相关与真值组内恢复误差权衡。
- 依据：三类Spearman相关矩阵。
- 观察：重点查看误差矩阵中的非对角结构。
- 限制：相关提示估计权衡，不单独证明其机制来源。
"""
    atomic_write_text(report, output_dir / "analysis-report.md")
    atomic_write_text(appendix, output_dir / "stats-appendix.md")
    atomic_write_text(catalog, output_dir / "figure-catalog.md")
    atomic_write_text(json.dumps(run_contract, ensure_ascii=False, indent=2, sort_keys=True), output_dir / "run-contract.json")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.workers, args.template_stride, args.maxiter, args.bootstrap_repeats) < 1:
        raise ValueError("workers, stride, maxiter and bootstrap repetitions must be positive")
    design_dir = args.design_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    design, truths, starts, empirical = validate_design(design_dir)
    mode = "formal" if args.max_datasets is None and args.template_stride == 1 else "test_only"
    run_contract = {
        "task_id": TASK_ID, "mode": mode, "design_contract_hash": design["contract_hash"],
        "design_contract_path": str(design_dir / "formal-design-contract.json"),
        "truth_bank_sha256": sha256_file(design_dir / "formal-truth-bank.csv"),
        "blind_start_bank_sha256": sha256_file(design_dir / "blind-start-bank.csv"),
        "runner_sha256": sha256_file(Path(__file__)), "template_stride": args.template_stride,
        "maxiter": args.maxiter, "bootstrap_repeats": args.bootstrap_repeats,
        "expected_formal_datasets": int(design["formal_datasets"]),
        "expected_formal_fits": int(design["formal_fits"]),
        "actual_template_rule": "40_scheduled_trials_with_device_valid_fixation_and_picture_timestamps_and_trial_specific_emotion_rectangles",
        "computational_backend": "exact_vectorized_loglikelihood_with_analytic_gradient",
    }
    contract_path = output_dir / "run-contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        fixed_keys = ("mode", "design_contract_hash", "truth_bank_sha256", "blind_start_bank_sha256", "template_stride", "maxiter")
        mismatched = [key for key in fixed_keys if previous.get(key) != run_contract.get(key)]
        if mismatched:
            raise ValueError(f"Resume contract mismatch: {mismatched}")
        if previous.get("runner_sha256") != run_contract.get("runner_sha256"):
            run_contract["runner_sha256_history"] = list(dict.fromkeys([
                *previous.get("runner_sha256_history", []), previous.get("runner_sha256"), run_contract["runner_sha256"],
            ]))
            run_contract["resume_computational_revision"] = (
                "Preserved completed checkpoints; unchanged normalized likelihood, bounds, starts and L-BFGS-B objective; "
                "subsequent fits use the verified exact analytic gradient."
            )
    atomic_write_text(json.dumps(run_contract, ensure_ascii=False, indent=2, sort_keys=True), contract_path)

    repeats = int(design["repeats_per_truth"])
    planned = [(str(row.truth_id), repeat) for row in truths.itertuples(index=False) for repeat in range(1, repeats + 1)]
    if args.max_datasets is not None:
        planned = planned[:args.max_datasets]
    planned_set = set(planned)
    completed_set = {
        (path.name.split("__R")[0], int(path.stem.split("__R")[1]))
        for path in checkpoint_dir.glob("truth_*__R*.json")
    }
    pending = planned_set - completed_set
    if not args.analysis_only and pending:
        truth_records = {str(row["truth_id"]): row for row in truths.to_dict("records")}
        start_records = starts.to_dict("records")
        payloads = []
        for truth_id in sorted({item[0] for item in pending}):
            truth_repeats = sorted(repeat for candidate, repeat in pending if candidate == truth_id)
            payloads.append({
                "truth": truth_records[truth_id], "starts": start_records, "repeats": truth_repeats,
                "checkpoint_dir": str(checkpoint_dir), "template_stride": args.template_stride,
                "base_seed": int(design["random_seed"]), "design_hash": design["contract_hash"],
                "fit_bounds": design["fit_bounds"], "maxiter": args.maxiter,
            })
        effective_workers = min(args.workers, len(payloads))
        if effective_workers == 1:
            for payload in payloads:
                result = run_truth_batch(payload)
                print(f"[T6 formal] completed batch {result}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                futures = [executor.submit(run_truth_batch, payload) for payload in payloads]
                for future in as_completed(futures):
                    result = future.result()
                    print(f"[T6 formal] completed batch {result}", flush=True)

    manifest, fits, best = load_checkpoints(checkpoint_dir)
    if not manifest.empty:
        selected = manifest.apply(lambda row: (str(row["truth_id"]), int(row["repeat"])) in planned_set, axis=1)
        manifest = manifest.loc[selected].sort_values(["truth_id", "repeat"]).reset_index(drop=True)
        fits = fits.loc[fits.apply(lambda row: (str(row["truth_id"]), int(row["repeat"])) in planned_set, axis=1)].sort_values(["truth_id", "repeat", "start_id"]).reset_index(drop=True)
        best = best.loc[best.apply(lambda row: (str(row["truth_id"]), int(row["repeat"])) in planned_set, axis=1)].sort_values(["truth_id", "repeat"]).reset_index(drop=True)
    if not manifest.empty:
        observed_trial_map = {}
        for truth in truths.to_dict("records"):
            template = load_actual_template(truth, args.template_stride)
            observed_trial_map[str(truth["truth_id"])] = int(template["observed_trials"][0])
        manifest["scheduled_trials"] = 40
        manifest["observed_trials"] = manifest["truth_id"].map(observed_trial_map).astype(int)
    atomic_write_csv(manifest, output_dir / "simulation-manifest.csv")
    atomic_write_csv(fits, output_dir / "all-fit-results.csv")
    atomic_write_csv(best, output_dir / "best-recovery-results.csv")
    progress = pd.DataFrame([{
        "mode": mode, "datasets_planned": len(planned_set), "datasets_completed": len(best),
        "fits_completed": len(fits), "complete": len(best) == len(planned_set),
        "design_contract_hash": design["contract_hash"],
    }])
    atomic_write_csv(progress, output_dir / "run-progress.csv")
    if len(best) != len(planned_set):
        return {"status": "IN_PROGRESS", "datasets_completed": len(best), "datasets_planned": len(planned_set), "output_dir": str(output_dir)}
    if mode != "formal":
        return {"status": "TEST_ONLY_COMPLETE", "datasets_completed": len(best), "fits": len(fits), "output_dir": str(output_dir)}

    summary, per_truth, matrices, implementation = summarize_results(
        best, fits, manifest, truths, empirical, args.bootstrap_repeats, int(design["random_seed"]),
    )
    optimizer_frame = manifest.copy()
    optimizer_frame["likelihood_gap"] = optimizer_frame["best_log_likelihood"] - optimizer_frame["truth_log_likelihood"]
    optimizer_by_truth = optimizer_frame.groupby("truth_id", as_index=False).agg(
        datasets=("dataset_id", "size"),
        best_not_below_truth_fraction=("best_not_below_truth", "mean"),
        likelihood_gap_mean=("likelihood_gap", "mean"),
        likelihood_gap_median=("likelihood_gap", "median"),
        likelihood_gap_min=("likelihood_gap", "min"),
        successful_fits_min=("successful_fits", "min"),
    )
    pm_rows = []
    for row in best.to_dict("records"):
        truth_pm = np.asarray([row[f"truth_pm_{emotion}"] for emotion in EMOTIONS], dtype=float)
        fitted_pm = np.asarray([row[f"fitted_pm_{emotion}"] for emotion in EMOTIONS], dtype=float)
        truth_clr = np.log(truth_pm) - np.log(truth_pm).mean()
        fitted_clr = np.log(fitted_pm) - np.log(fitted_pm).mean()
        pm_rows.append({
            "dataset_id": row["dataset_id"], "truth_id": row["truth_id"],
            "aitchison_distance": float(np.linalg.norm(fitted_clr - truth_clr)),
            "total_variation_distance": float(0.5 * np.abs(fitted_pm - truth_pm).sum()),
        })
    pm_composition = pd.DataFrame(pm_rows)
    pm_composition_summary = pd.DataFrame([{
        "datasets": len(pm_composition),
        "aitchison_distance_mean": float(pm_composition["aitchison_distance"].mean()),
        "aitchison_distance_median": float(pm_composition["aitchison_distance"].median()),
        "aitchison_distance_p95": float(pm_composition["aitchison_distance"].quantile(0.95)),
        "aitchison_distance_max": float(pm_composition["aitchison_distance"].max()),
        "total_variation_distance_mean": float(pm_composition["total_variation_distance"].mean()),
        "total_variation_distance_median": float(pm_composition["total_variation_distance"].median()),
        "total_variation_distance_p95": float(pm_composition["total_variation_distance"].quantile(0.95)),
        "total_variation_distance_max": float(pm_composition["total_variation_distance"].max()),
    }])
    pm_composition_by_truth = pm_composition.groupby("truth_id", as_index=False).agg(
        datasets=("dataset_id", "size"),
        aitchison_distance_mean=("aitchison_distance", "mean"),
        aitchison_distance_median=("aitchison_distance", "median"),
        aitchison_distance_p95=("aitchison_distance", lambda values: values.quantile(0.95)),
        total_variation_distance_mean=("total_variation_distance", "mean"),
        total_variation_distance_median=("total_variation_distance", "median"),
        total_variation_distance_p95=("total_variation_distance", lambda values: values.quantile(0.95)),
    )
    atomic_write_csv(summary, output_dir / "parameter-summary.csv")
    atomic_write_csv(per_truth, output_dir / "per-truth-performance.csv")
    atomic_write_csv(implementation, output_dir / "implementation-summary.csv")
    atomic_write_csv(optimizer_by_truth, output_dir / "optimizer-by-truth.csv")
    atomic_write_csv(pm_composition, output_dir / "pm-compositional-distance.csv")
    atomic_write_csv(pm_composition_summary, output_dir / "pm-compositional-summary.csv")
    atomic_write_csv(pm_composition_by_truth, output_dir / "pm-compositional-by-truth.csv")
    for name, matrix in matrices.items():
        atomic_write_csv(matrix.rename_axis("parameter").reset_index(), output_dir / f"{name}.csv")
    make_figures(per_truth, summary, matrices, output_dir)
    run_contract["formal_outcomes_generated"] = True
    run_contract["completed_datasets"] = len(best)
    run_contract["completed_fits"] = len(fits)
    render_reports(
        summary, implementation, manifest, matrices, optimizer_by_truth,
        pm_composition_summary, run_contract, output_dir,
    )
    return {"status": "FORMAL_ANALYSIS_COMPLETE_NEEDS_REVIEW", "datasets_completed": len(best), "fits": len(fits), "output_dir": str(output_dir)}


if __name__ == "__main__":
    print(run(parse_args()))
