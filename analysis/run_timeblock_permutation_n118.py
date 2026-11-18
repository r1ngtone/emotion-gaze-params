#!/usr/bin/env python3
"""Shared T5/T6 held-out time-block permutation validation.

Both tasks use the same contract: four-fold trial-wise cross-validation,
250-ms within-trial block permutation of complete spatial observations,
1,000 shuffled test trajectories, and participant-level inference.

T5 is scored with its formal spatial log-likelihood. T6 is scored with its
current empirical objective. The shared result is therefore called a held-out
model-score advantage rather than a cross-task likelihood difference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-t5t6-timeblock")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cache-t5t6-timeblock")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402
from scipy.optimize import minimize  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[6]
T5_ROOT = REPO_ROOT / "single_pic" / "单图" / "t5_divide"
T6_ROOT = REPO_ROOT / "multipic_eye" / "多图" / "t6_devide"
for path in (T5_ROOT / "src", T5_ROOT, T6_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from t5_memory.model import (  # noqa: E402
    load_a5_dataframe,
    loglik_picture_only as official_t5_score,
)
from t5_memory.pipeline import run_memory_pipeline_from_dataframes  # noqa: E402
from corecode.pipeline import (  # noqa: E402
    annotate_spatial_and_emotion,
    build_base_frame,
    compute_log_likelihood as score_t6,
    optimize_free_parameters as fit_t6,
    prepare_model_data,
)


TASK_ID = "T20260727-001-t5-t6-likelihood-permutation"
INPUT_MANIFEST = (
    REPO_ROOT
    / "t5t6 联合数据分析/260616/data/20260624_t5_t6_118_rerun/input_manifest.csv"
)
DEFAULT_CHECKPOINT_DIR = Path("/private/tmp/t5t6_timeblock_permutation_checkpoints")
TASKS = ("T5", "T6")
EMOTIONS_T5 = ("positive", "negative", "neutral", "complex")
FOLDS = 4
BLOCK_MS = 250.0
FORMAL_PERMUTATIONS = 1000
FORMAL_SEED = 20260727
BOOTSTRAP_REPEATS = 10000
SIGN_FLIP_REPEATS = 20000
T5_RESAMPLE_RATIO = 0.07


@dataclass(frozen=True)
class SubjectInput:
    subject_id: str
    name: str
    gender: str
    t5_eyetrack: str
    t5_records: str
    t6_eyetrack: str
    t6_records: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--permutations", type=int, default=FORMAL_PERMUTATIONS)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--keep-checkpoints", action="store_true")
    return parser.parse_args()


def stable_seed(base: int, subject_id: str, task: str, fold: int) -> int:
    text = f"{base}|{subject_id}|{task}|{fold}"
    value = 2166136261
    for byte in text.encode("utf-8"):
        value = (value ^ byte) * 16777619 & 0xFFFFFFFF
    return int(value)


def read_subjects(manifest_path: Path, max_subjects: int | None) -> list[SubjectInput]:
    frame = pd.read_csv(manifest_path, dtype={"subject_id": str})
    required = {"subject_id", "name", "gender", "task_type", "eyetrack_path", "records_path", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    frame = frame.loc[frame["status"].eq("success")].copy()
    rows: list[SubjectInput] = []
    for subject_id, group in frame.groupby("subject_id", sort=True):
        task_map = {str(row.task_type): row for row in group.itertuples(index=False)}
        if set(task_map) != set(TASKS):
            continue
        t5 = task_map["T5"]
        t6 = task_map["T6"]
        paths = [t5.eyetrack_path, t5.records_path, t6.eyetrack_path, t6.records_path]
        if not all(Path(str(path)).is_file() for path in paths):
            raise FileNotFoundError(f"Missing input for subject {subject_id}: {paths}")
        rows.append(
            SubjectInput(
                subject_id=str(subject_id), name=str(t5.name), gender=str(t5.gender),
                t5_eyetrack=str(t5.eyetrack_path), t5_records=str(t5.records_path),
                t6_eyetrack=str(t6.eyetrack_path), t6_records=str(t6.records_path),
            )
        )
    if max_subjects is not None:
        rows = rows[:max_subjects]
    return rows


def t5_data_from_paths(eyetrack_path: str, records_path: str) -> dict[str, Any]:
    eyetrack = pd.read_csv(eyetrack_path, encoding="utf-8-sig")
    records = pd.read_csv(records_path, encoding="utf-8-sig")
    a5 = run_memory_pipeline_from_dataframes(eyetrack, records)["A5"]
    return load_a5_dataframe(
        a5, timestamp_unit_scale=1e9, x_min=480, y_min=180,
        width=960, height=720, resample_ratio=T5_RESAMPLE_RATIO,
    )


def t6_data_from_paths(eyetrack_path: str, records_path: str) -> dict[str, np.ndarray]:
    eyetrack = pd.read_csv(eyetrack_path, encoding="utf-8-sig")
    records = pd.read_csv(records_path, encoding="utf-8-sig")
    base, _ = build_base_frame(eyetrack)
    annotated = annotate_spatial_and_emotion(base, records)
    _, data = prepare_model_data(annotated, timestamp_unit_scale=1e6)
    return data


def t5_fold_map(data: Mapping[str, Any]) -> dict[int, int]:
    by_emotion: dict[str, list[int]] = {emotion: [] for emotion in EMOTIONS_T5}
    for trial in data["trials"]:
        by_emotion[str(trial["emotion"])].append(int(trial["trial_number"]))
    mapping: dict[int, int] = {}
    for emotion in EMOTIONS_T5:
        trials = sorted(set(by_emotion[emotion]))
        if len(trials) < FOLDS:
            raise ValueError(f"T5 emotion {emotion} has fewer than {FOLDS} trials")
        for index, trial in enumerate(trials):
            mapping[trial] = index % FOLDS
    return mapping


def t6_fold_map(data: Mapping[str, np.ndarray]) -> dict[int, int]:
    trials = sorted(int(value) for value in np.unique(data["trial_number"]) if int(value) > 0)
    if len(trials) < FOLDS:
        raise ValueError("T6 has fewer trials than folds")
    return {trial: index % FOLDS for index, trial in enumerate(trials)}


def subset_t5(data: Mapping[str, Any], mapping: Mapping[int, int], fold: int, test: bool) -> dict[str, Any]:
    trials = [
        trial for trial in data["trials"]
        if (mapping[int(trial["trial_number"])] == fold) == test
    ]
    return {**{key: value for key, value in data.items() if key != "trials"}, "trials": trials}


def subset_t6(data: Mapping[str, np.ndarray], mapping: Mapping[int, int], fold: int, test: bool) -> dict[str, np.ndarray]:
    fold_values = np.asarray([mapping.get(int(trial), -1) for trial in data["trial_number"]])
    mask = (fold_values == fold) if test else (fold_values != fold)
    return {key: np.asarray(value)[mask] for key, value in data.items()}


def block_permutation_indices(times: np.ndarray, block_width: float, rng: np.random.Generator) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if len(times) < 2:
        return np.arange(len(times), dtype=int)
    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    block_ids = np.floor((sorted_times - sorted_times[0]) / block_width).astype(int)
    blocks = [order[block_ids == block] for block in np.unique(block_ids)]
    if len(blocks) < 2:
        return order
    block_order = rng.permutation(len(blocks))
    if np.array_equal(block_order, np.arange(len(blocks))):
        block_order = np.roll(block_order, 1)
    return np.concatenate([blocks[int(index)] for index in block_order])


def permute_t5_test(data: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    trials = []
    for trial in data["trials"]:
        indices = block_permutation_indices(np.asarray(trial["t"]), BLOCK_MS / 1000.0, rng)
        copied = dict(trial)
        copied["x"] = np.asarray(trial["x"])[indices]
        copied["y"] = np.asarray(trial["y"])[indices]
        copied["t"] = np.sort(np.asarray(trial["t"], dtype=float), kind="stable")
        trials.append(copied)
    return {**{key: value for key, value in data.items() if key != "trials"}, "trials": trials}


def permute_t6_test(data: Mapping[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    output = {key: np.asarray(value).copy() for key, value in data.items()}
    spatial_keys = ("gaze_x", "gaze_y", "positive", "nagetive", "neutral")
    for trial in np.unique(data["trial_number"]):
        positions = np.flatnonzero(data["trial_number"] == trial)
        indices_local = block_permutation_indices(
            np.asarray(data["t_from_trial_start"])[positions], BLOCK_MS, rng,
        )
        source = positions[indices_local]
        for key in spatial_keys:
            output[key][positions] = np.asarray(data[key])[source]
    return output


@njit(cache=True, nogil=True)
def t5_trial_score_numba(
    x: np.ndarray, y: np.ndarray, t: np.ndarray,
    tau: float, sigma_s: float, dc: float,
    x_centers: np.ndarray, y_centers: np.ndarray,
) -> float:
    cells = len(x_centers)
    attention = np.ones(cells, dtype=np.float64)
    total = 0.0
    last_t = 0.0
    for index in range(len(x)):
        if index > 0:
            dt = max(0.0, t[index] - last_t)
            alpha = math.exp(-dt / tau)
            for cell in range(cells):
                attention[cell] *= alpha
        attention_sum = 0.0
        for cell in range(cells):
            attention_sum += attention[cell]
        probability = 0.0
        denominator = 2.0 * math.pi * dc * dc
        for cell in range(cells):
            dx = x_centers[cell] - x[index]
            dy = y_centers[cell] - y[index]
            gaussian = math.exp(-(dx * dx + dy * dy) / (2.0 * dc * dc)) / denominator
            probability += attention[cell] * gaussian / max(attention_sum, 1e-300)
        total += math.log(max(probability, 1e-300))
        for cell in range(cells):
            dx = x_centers[cell] - x[index]
            dy = y_centers[cell] - y[index]
            attention[cell] += math.exp(-(dx * dx + dy * dy) / (2.0 * sigma_s * sigma_s))
        last_t = t[index]
    return total


def t5_fast_score(data: Mapping[str, Any], theta: np.ndarray) -> float:
    xs = (np.arange(12) + 0.5) * (float(data["width"]) / 12)
    ys = (np.arange(9) + 0.5) * (float(data["height"]) / 9)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    x_centers = x_grid.ravel()
    y_centers = y_grid.ravel()
    emotion_index = {emotion: index for index, emotion in enumerate(EMOTIONS_T5)}
    total = 0.0
    for trial in data["trials"]:
        offset = emotion_index[str(trial["emotion"])] * 3
        total += t5_trial_score_numba(
            np.asarray(trial["x"], dtype=float), np.asarray(trial["y"], dtype=float),
            np.asarray(trial["t"], dtype=float), float(theta[offset]),
            float(theta[offset + 1]), float(theta[offset + 2]), x_centers, y_centers,
        )
    return float(total)


@njit(cache=True, nogil=True)
def t6_score_numba(
    x: np.ndarray, y: np.ndarray, positive: np.ndarray,
    negative: np.ndarray, neutral: np.ndarray, t_picture: np.ndarray,
    tmax1: float, tmax2: float, tmax3: float,
    pmax1: float, pmax2: float, pmax3: float, dc: float,
) -> float:
    eps = 1e-12
    total = 0.0
    gaussian_denominator = 2.0 * math.pi * dc * dc
    gaussian_exponent_denominator = 2.0 * dc * dc + eps
    for index in range(len(x)):
        time_value = t_picture[index]
        raw1 = 0.0
        raw2 = 0.0
        raw3 = 0.0
        if time_value >= 0.0:
            raw1 = pmax1 * (time_value / (tmax1 + eps)) * math.exp(1.0 - time_value / (tmax1 + eps))
            raw2 = pmax2 * (time_value / (tmax2 + eps)) * math.exp(1.0 - time_value / (tmax2 + eps))
            raw3 = pmax3 * (time_value / (tmax3 + eps)) * math.exp(1.0 - time_value / (tmax3 + eps))
        raw_sum = 1.0 + raw1 + raw2 + raw3
        p0 = 1.0 / (raw_sum + eps)
        p1 = raw1 / (raw_sum + eps)
        p2 = raw2 / (raw_sum + eps)
        p3 = raw3 / (raw_sum + eps)
        dx = x[index] - 960.0
        dy = y[index] - 540.0
        g0 = math.exp(-(dx * dx + dy * dy) / gaussian_exponent_denominator) / gaussian_denominator
        value = p0 * g0 + p1 * positive[index] + p2 * negative[index] + p3 * neutral[index]
        total += math.log(max(value, eps))
    return total


def t6_fast_score(data: Mapping[str, np.ndarray], params: Mapping[str, float]) -> float:
    return float(t6_score_numba(
        np.asarray(data["gaze_x"], dtype=float), np.asarray(data["gaze_y"], dtype=float),
        np.asarray(data["positive"], dtype=float), np.asarray(data["nagetive"], dtype=float),
        np.asarray(data["neutral"], dtype=float), np.asarray(data["t_from_picture_start"], dtype=float),
        float(params["Tmax1"]), float(params["Tmax2"]), float(params["Tmax3"]),
        float(params["Pmax1"]), float(params["Pmax2"]), float(params["Pmax3"]),
        float(params["dc"]),
    ))


def validate_t6_fast_score(data: Mapping[str, np.ndarray], params: Mapping[str, float]) -> float:
    official = float(score_t6(data, params))
    fast = t6_fast_score(data, params)
    tolerance = 1e-8 * max(1.0, abs(official))
    if abs(official - fast) > tolerance:
        raise ValueError(f"T6 fast-score mismatch: official={official}, fast={fast}")
    return fast


def fit_t5_separable(data: Mapping[str, Any]) -> dict[str, Any]:
    """Fit the four independent T5 emotion blocks with the formal contract."""
    theta = np.empty(12, dtype=float)
    successes: list[bool] = []
    messages: list[str] = []
    for emotion_index, emotion in enumerate(EMOTIONS_T5):
        emotion_data = {
            **{key: value for key, value in data.items() if key != "trials"},
            "trials": [trial for trial in data["trials"] if str(trial["emotion"]) == emotion],
        }
        if not emotion_data["trials"]:
            raise ValueError(f"No T5 training trials for {emotion}")

        def objective(values: np.ndarray) -> float:
            local_theta = np.tile(np.asarray(values, dtype=float), 4)
            return -t5_fast_score(emotion_data, local_theta)

        tau_low = 0.0001 if emotion == "complex" else 0.01
        result = minimize(
            objective, x0=np.asarray([0.50, 120.0, 80.0]),
            bounds=[(tau_low, 10.0), (5.0, 600.0), (5.0, 400.0)],
            method="L-BFGS-B",
        )
        if not result.success:
            result = minimize(
                objective, x0=np.asarray(result.x, dtype=float),
                bounds=[(tau_low, 10.0), (5.0, 600.0), (5.0, 400.0)],
                method="L-BFGS-B", options={"maxls": 100},
            )
        offset = emotion_index * 3
        theta[offset:offset + 3] = result.x
        successes.append(bool(result.success))
        messages.append(f"{emotion}:{result.message}")
    return {"theta_opt": theta, "success": all(successes), "message": "; ".join(messages)}


def prepare_t5_reference_inputs(data: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    xs = (np.arange(12) + 0.5) * (float(data["width"]) / 12)
    ys = (np.arange(9) + 0.5) * (float(data["height"]) / 9)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    x_centers = x_grid.ravel()
    y_centers = y_grid.ravel()
    offsets = [0]
    emotion_indices: list[int] = []
    times: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    emotion_index = {emotion: index for index, emotion in enumerate(EMOTIONS_T5)}
    for trial in data["trials"]:
        x = np.asarray(trial["x"], dtype=float)
        y = np.asarray(trial["y"], dtype=float)
        times.append(np.asarray(trial["t"], dtype=float))
        distances.append((x[:, None] - x_centers[None, :]) ** 2 + (y[:, None] - y_centers[None, :]) ** 2)
        offsets.append(offsets[-1] + len(x))
        emotion_indices.append(emotion_index[str(trial["emotion"])])
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(emotion_indices, dtype=np.int64),
        np.concatenate(times).astype(np.float64),
        np.ascontiguousarray(np.vstack(distances), dtype=np.float64),
    )


@njit(cache=True, nogil=True)
def t5_reference_compiled_numba(
    theta: np.ndarray, offsets: np.ndarray, emotion_indices: np.ndarray,
    t_all: np.ndarray, dist2: np.ndarray,
) -> float:
    eps = 1e-300
    total_loglik = 0.0
    grid_count = dist2.shape[1]
    two_pi = 2.0 * math.pi
    for trial_index in range(len(emotion_indices)):
        parameter_start = emotion_indices[trial_index] * 3
        tau = theta[parameter_start]
        sigma_s = theta[parameter_start + 1]
        dc = theta[parameter_start + 2]
        attention = np.ones(grid_count, dtype=np.float64)
        start = offsets[trial_index]
        stop = offsets[trial_index + 1]
        last_t = 0.0
        has_last_t = False
        for sample_index in range(start, stop):
            current_t = t_all[sample_index]
            if has_last_t:
                dt = current_t - last_t
                if dt < 0.0:
                    dt = 0.0
                attention *= math.exp(-dt / tau)
            attention_sum = attention.sum()
            if attention_sum < eps:
                attention_sum = eps
            gaussian_scale = two_pi * dc * dc
            p_xy = 0.0
            for grid_index in range(grid_count):
                gaussian = math.exp(-dist2[sample_index, grid_index] / (2.0 * dc * dc)) / gaussian_scale
                p_xy += (attention[grid_index] / attention_sum) * gaussian
            if p_xy < eps:
                p_xy = eps
            total_loglik += math.log(p_xy)
            for grid_index in range(grid_count):
                attention[grid_index] += math.exp(
                    -dist2[sample_index, grid_index] / (2.0 * sigma_s * sigma_s)
                )
            last_t = current_t
            has_last_t = True
    return total_loglik


def fit_t5_separable_reference(data: Mapping[str, Any]) -> dict[str, Any]:
    """Refit with the formal compiled likelihood when the fast scorer diverges."""
    theta = np.empty(12, dtype=float)
    successes: list[bool] = []
    messages: list[str] = []
    for emotion_index, emotion in enumerate(EMOTIONS_T5):
        emotion_data = {
            **{key: value for key, value in data.items() if key != "trials"},
            "trials": [trial for trial in data["trials"] if str(trial["emotion"]) == emotion],
        }
        compiled_inputs = prepare_t5_reference_inputs(emotion_data)

        def objective(values: np.ndarray) -> float:
            local_theta = np.tile(np.asarray(values, dtype=float), 4)
            return -float(t5_reference_compiled_numba(local_theta, *compiled_inputs))

        tau_low = 0.0001 if emotion == "complex" else 0.01
        result = minimize(
            objective, x0=np.asarray([0.50, 120.0, 80.0]),
            bounds=[(tau_low, 10.0), (5.0, 600.0), (5.0, 400.0)],
            method="L-BFGS-B",
        )
        if not result.success:
            result = minimize(
                objective, x0=np.asarray(result.x, dtype=float),
                bounds=[(tau_low, 10.0), (5.0, 600.0), (5.0, 400.0)],
                method="L-BFGS-B", options={"maxls": 100},
            )
        offset = emotion_index * 3
        theta[offset:offset + 3] = result.x
        successes.append(bool(result.success))
        messages.append(f"{emotion}:{result.message}")
    return {"theta_opt": theta, "success": all(successes), "message": "; ".join(messages)}


def t5_reference_score(data: Mapping[str, Any], theta: np.ndarray) -> float:
    return float(t5_reference_compiled_numba(np.asarray(theta, dtype=float), *prepare_t5_reference_inputs(data)))


def t5_reference_permuted_score(
    compiled_inputs: tuple[np.ndarray, ...], theta: np.ndarray, rng: np.random.Generator,
) -> float:
    offsets, emotion_indices, t_all, dist2 = compiled_inputs
    shuffled_dist2 = np.empty_like(dist2)
    for trial_index in range(len(emotion_indices)):
        start = int(offsets[trial_index])
        stop = int(offsets[trial_index + 1])
        indices = block_permutation_indices(t_all[start:stop], BLOCK_MS / 1000.0, rng)
        shuffled_dist2[start:stop] = dist2[start:stop][indices]
    return float(t5_reference_compiled_numba(theta, offsets, emotion_indices, t_all, shuffled_dist2))


def validate_t5_fast_score(data: Mapping[str, Any], theta: np.ndarray) -> float:
    official = float(official_t5_score(data, theta, M=12, N=9))
    fast = t5_fast_score(data, theta)
    tolerance = 1e-8 * max(1.0, abs(official))
    if abs(official - fast) > tolerance:
        raise ValueError(f"T5 fast-score mismatch: official={official}, fast={fast}")
    return fast


def observation_count_t5(data: Mapping[str, Any]) -> int:
    return int(sum(len(trial["x"]) for trial in data["trials"]))


def fit_and_score_t5(
    data: Mapping[str, Any], subject_id: str, permutations: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = t5_fold_map(data)
    rows: list[dict[str, Any]] = []
    real_total = 0.0
    null_total = np.zeros(permutations, dtype=float)
    observations_total = 0
    for fold in range(FOLDS):
        train = subset_t5(data, mapping, fold, test=False)
        test = subset_t5(data, mapping, fold, test=True)
        fit = fit_t5_separable(train)
        theta = np.asarray(fit["theta_opt"], dtype=float)
        use_reference = False
        try:
            real = validate_t5_fast_score(test, theta)
        except ValueError:
            fit = fit_t5_separable_reference(train)
            theta = np.asarray(fit["theta_opt"], dtype=float)
            real = t5_reference_score(test, theta)
            official = float(official_t5_score(test, theta, M=12, N=9))
            tolerance = 1e-8 * max(1.0, abs(official))
            if abs(official - real) > tolerance:
                raise ValueError(f"T5 reference-score mismatch: official={official}, compiled={real}")
            use_reference = True
        observations = observation_count_t5(test)
        rng = np.random.default_rng(stable_seed(seed, subject_id, "T5", fold))
        null = np.empty(permutations, dtype=float)
        reference_inputs = prepare_t5_reference_inputs(test) if use_reference else None
        for permutation in range(permutations):
            null[permutation] = (
                t5_reference_permuted_score(reference_inputs, theta, rng)
                if use_reference
                else t5_fast_score(permute_t5_test(test, rng), theta)
            )
        real_total += real
        null_total += null
        observations_total += observations
        rows.append({
            "subject_id": subject_id, "task": "T5", "fold": fold + 1,
            "train_trials": len(train["trials"]), "test_trials": len(test["trials"]),
            "test_observations": observations, "fit_success": bool(fit["success"]),
            "fit_message": str(fit["message"]), "real_score": real,
            "shuffle_score_mean": float(null.mean()),
            "delta_score_per_observation": float((real - null.mean()) / observations),
        })
    return rows, subject_summary(subject_id, "T5", real_total, null_total, observations_total)


def fit_and_score_t6(
    data: Mapping[str, np.ndarray], subject_id: str, permutations: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = t6_fold_map(data)
    rows: list[dict[str, Any]] = []
    real_total = 0.0
    null_total = np.zeros(permutations, dtype=float)
    observations_total = 0
    for fold in range(FOLDS):
        train = subset_t6(data, mapping, fold, test=False)
        test = subset_t6(data, mapping, fold, test=True)
        fit = fit_t6(train)
        if not fit["success"]:
            retry = fit_t6(train, init_params=fit["best_params"], method="Powell")
            if retry["success"] or retry["best_log_likelihood"] >= fit["best_log_likelihood"]:
                fit = retry
        params = fit["best_params"]
        real = validate_t6_fast_score(test, params)
        observations = int(len(test["timestamp"]))
        rng = np.random.default_rng(stable_seed(seed, subject_id, "T6", fold))
        null = np.empty(permutations, dtype=float)
        for permutation in range(permutations):
            null[permutation] = t6_fast_score(permute_t6_test(test, rng), params)
        real_total += real
        null_total += null
        observations_total += observations
        rows.append({
            "subject_id": subject_id, "task": "T6", "fold": fold + 1,
            "train_trials": int(np.unique(train["trial_number"]).size),
            "test_trials": int(np.unique(test["trial_number"]).size),
            "test_observations": observations, "fit_success": bool(fit["success"]),
            "fit_message": str(fit["message"]), "real_score": real,
            "shuffle_score_mean": float(null.mean()),
            "delta_score_per_observation": float((real - null.mean()) / observations),
        })
    return rows, subject_summary(subject_id, "T6", real_total, null_total, observations_total)


def subject_summary(
    subject_id: str, task: str, real_total: float,
    null_total: np.ndarray, observations: int,
) -> dict[str, Any]:
    null_per_observation = null_total / observations
    real_per_observation = real_total / observations
    delta = real_per_observation - float(null_per_observation.mean())
    return {
        "subject_id": subject_id, "task": task, "test_observations": observations,
        "real_score_per_observation": real_per_observation,
        "shuffle_score_per_observation_mean": float(null_per_observation.mean()),
        "shuffle_score_per_observation_sd": float(null_per_observation.std(ddof=1)),
        "shuffle_score_per_observation_q025": float(np.quantile(null_per_observation, 0.025)),
        "shuffle_score_per_observation_q975": float(np.quantile(null_per_observation, 0.975)),
        "delta_score_per_observation": delta,
        "empirical_one_sided_p": float((np.sum(null_total >= real_total) + 1) / (len(null_total) + 1)),
    }


def run_subject(subject: SubjectInput, permutations: int, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        t5_data = t5_data_from_paths(subject.t5_eyetrack, subject.t5_records)
        t5_folds, t5_summary = fit_and_score_t5(t5_data, subject.subject_id, permutations, seed)
        t6_data = t6_data_from_paths(subject.t6_eyetrack, subject.t6_records)
        t6_folds, t6_summary = fit_and_score_t6(t6_data, subject.subject_id, permutations, seed)
        for row in (t5_summary, t6_summary):
            row.update({"name": subject.name, "gender": subject.gender})
        return {
            "status": "success", "subject_id": subject.subject_id,
            "folds": [*t5_folds, *t6_folds], "subjects": [t5_summary, t6_summary],
            "runtime_seconds": time.perf_counter() - started,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed", "subject_id": subject.subject_id,
            "error_type": type(exc).__name__, "error_message": str(exc),
            "traceback": traceback.format_exc(), "runtime_seconds": time.perf_counter() - started,
        }


def checkpoint_file(checkpoint_dir: Path, subject_id: str) -> Path:
    return checkpoint_dir / f"{subject_id}.json"


def write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_completed(checkpoint_dir: Path, subject_id: str) -> dict[str, Any] | None:
    path = checkpoint_file(checkpoint_dir, subject_id)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("status") == "success" else None


def bootstrap_ci(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for index in range(repeats):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_p(values: np.ndarray, repeats: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    hits = 0
    batch = 2000
    done = 0
    while done < repeats:
        size = min(batch, repeats - done)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(values)))
        hits += int(np.sum((signs * values).mean(axis=1) >= observed))
        done += size
    return float((hits + 1) / (repeats + 1))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def summarize(subject_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    interim: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for task in TASKS:
        values = subject_frame.loc[subject_frame["task"].eq(task), "delta_score_per_observation"].to_numpy(float)
        low, high = bootstrap_ci(values, BOOTSTRAP_REPEATS, seed + (1 if task == "T5" else 2))
        p_value = sign_flip_p(values, SIGN_FLIP_REPEATS, seed + (11 if task == "T5" else 12))
        raw_p[task] = p_value
        interim.append({
            "task": task, "n_subjects": len(values), "mean_delta": float(values.mean()),
            "sd_delta": float(values.std(ddof=1)), "median_delta": float(np.median(values)),
            "iqr_delta": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
            "paired_standardized_effect_dz": float(values.mean() / values.std(ddof=1)),
            "n_positive": int(np.sum(values > 0)), "proportion_positive": float(np.mean(values > 0)),
            "sign_flip_one_sided_p": p_value,
        })
    adjusted = holm_adjust(raw_p)
    for row in interim:
        row["holm_adjusted_p"] = adjusted[str(row["task"])]
        row["passes_prespecified_gate"] = bool(
            row["mean_delta"] > 0 and row["bootstrap_ci95_low"] > 0 and row["holm_adjusted_p"] < 0.05
        )
    return pd.DataFrame(interim)


def save_figures(subjects: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 160, "savefig.dpi": 300,
    })
    colors = {"T5": "#0072B2", "T6": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), constrained_layout=True)
    rng = np.random.default_rng(FORMAL_SEED + 99)
    panel_titles = {
        "T5": "T5 slow single-image task",
        "T6": "T6 rapid multi-image task",
    }
    axis_labels = {
        "T5": "Held-out log-likelihood advantage\n(real − shuffled, per observation)",
        "T6": "Held-out empirical objective-score advantage\n(real − shuffled, per observation)",
    }
    for panel_index, (ax, task) in enumerate(zip(axes, TASKS)):
        values = subjects.loc[subjects["task"].eq(task), "delta_score_per_observation"].to_numpy(float)
        jitter = np.clip(rng.normal(0.02, 0.075, size=len(values)), -0.16, 0.20)
        ax.scatter(values, jitter, s=20, alpha=0.52, color=colors[task], edgecolor="none", rasterized=True)
        row = summary.loc[summary["task"].eq(task)].iloc[0]
        mean_value = float(row["mean_delta"])
        ci_low = float(row["bootstrap_ci95_low"])
        ci_high = float(row["bootstrap_ci95_high"])
        ax.errorbar(
            [mean_value], [0.39],
            xerr=[[mean_value - ci_low], [ci_high - mean_value]],
            fmt="D", color="#111111", markerfacecolor="#111111",
            capsize=4, capthick=1.2, lw=1.5, ms=5.5, zorder=5,
        )
        ax.axvline(0, color="#444444", ls="--", lw=1.0, zorder=0)
        span = float(values.max() - min(0.0, values.min()))
        ax.set_xlim(-0.035 * span, float(values.max()) + 0.08 * span)
        ax.set_ylim(-0.24, 0.53)
        ax.set_yticks([])
        ax.set_xlabel(axis_labels[task], fontsize=9.2)
        ax.set_title(f"{chr(65 + panel_index)}  {panel_titles[task]}", loc="left", fontsize=10.2, fontweight="bold")
        ax.text(
            0.02, 0.92,
            f"N = {int(row['n_subjects'])}; {int(row['n_positive'])}/{int(row['n_subjects'])} > 0\n"
            f"Mean = {mean_value:.2f}  [95% CI {ci_low:.2f}, {ci_high:.2f}]",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color="#222222",
        )
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
        ax.spines["left"].set_visible(False)
    fig.suptitle("Held-out time-block permutation validation", fontsize=12.0, fontweight="bold", y=1.04)
    fig.text(
        0.5, -0.055,
        "Task-specific scales: compare each distribution with zero; do not compare absolute magnitudes across panels.",
        ha="center", va="top", fontsize=8.0, color="#444444",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"Figure_T5_T6_timeblock_permutation_summary_N118.{suffix}",
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_reports(subjects: pd.DataFrame, folds: pd.DataFrame, summary: pd.DataFrame, failures: pd.DataFrame, output_dir: Path, permutations: int) -> None:
    values = {row.task: row for row in summary.itertuples(index=False)}
    dual_pass = all(bool(values[task].passes_prespecified_gate) for task in TASKS)
    analysis = f"""# T5–T6时间块打乱验证分析报告

状态：needs_review

## 分析问题

把T5和T6视为一个双任务体系，在相同四折交叉验证和试次内250 ms时间块置换下，分别检验真实gaze轨迹的测试集模型分数是否高于打乱轨迹。

## 共同方法

- 样本：T5与T6各N={subjects['subject_id'].nunique()}。
- 每个任务均采用四折trial交叉验证。
- 每个测试trial按250 ms切块；块内完整观测顺序保留，时间块随机重排。
- 每名参与者、每个任务使用{permutations}套打乱轨迹。
- 总体推断单位为参与者；T5与T6分别检验，并对两个主要p值进行Holm校正。
- T5分数为正式log-likelihood；T6分数为当前经验目标函数。二者统一解释为held-out model-score advantage，不直接比较原始量纲。

## 主要结果

| Task | N | Mean delta | 95% CI | Positive | Holm p | Gate |
|---|---:|---:|---|---:|---:|---|
| T5 | {values['T5'].n_subjects} | {values['T5'].mean_delta:.8g} | [{values['T5'].bootstrap_ci95_low:.8g}, {values['T5'].bootstrap_ci95_high:.8g}] | {values['T5'].n_positive}/{values['T5'].n_subjects} | {values['T5'].holm_adjusted_p:.6g} | {values['T5'].passes_prespecified_gate} |
| T6 | {values['T6'].n_subjects} | {values['T6'].mean_delta:.8g} | [{values['T6'].bootstrap_ci95_low:.8g}, {values['T6'].bootstrap_ci95_high:.8g}] | {values['T6'].n_positive}/{values['T6'].n_subjects} | {values['T6'].holm_adjusted_p:.6g} | {values['T6'].passes_prespecified_gate} |

## 判定

双任务共同通过预设门槛：**{dual_pass}**。该检验回答模型分数是否依赖真实轨迹与任务时间轴的长期对应；它不替代参数恢复、model-free行为锚定或情绪条件敏感性验证。

## 边界

T6当前经验目标函数不是归一化空间likelihood，因此本文不把T6结果称为likelihood增益。T5与T6的原始分数不可横向比较，只比较各任务内部真实相对打乱的差值方向和不确定性。
"""
    (output_dir / "analysis-report.md").write_text(analysis, encoding="utf-8")

    stats = f"""# 统计附录

- Task ID: `{TASK_ID}`
- Folds: {FOLDS}
- Block width: {BLOCK_MS:.0f} ms
- Permutations per participant and task: {permutations}
- Bootstrap repeats: {BOOTSTRAP_REPEATS}
- Sign-flip repeats: {SIGN_FLIP_REPEATS}
- Family correction: Holm across T5 and T6
- Participant rows: {len(subjects)}
- Fold rows: {len(folds)}
- Failure rows: {len(failures)}

## 完整汇总

{dataframe_to_markdown(summary)}

## 预设通过标准

任务内平均delta为正、bootstrap 95% CI下界大于0、Holm校正单侧p小于0.05。双任务结论要求两项任务分别通过。
"""
    (output_dir / "stats-appendix.md").write_text(stats, encoding="utf-8")

    catalog = """# 图表目录

## Figure_T5_T6_timeblock_permutation_summary_N118

- 目的：用并列分面估计图展示两个任务各自的参与者差值分布、均值及bootstrap 95% CI。
- 图形：A面板为T5，B面板为T6；彩色点为参与者，黑色菱形和横向误差线为均值及bootstrap 95% CI，竖直虚线为零优势。
- 注意：两个面板使用任务特异的横轴和尺度，只判断各自分布相对零线的位置，不比较delta绝对大小。
- 决策：只有两个CI均位于零线上方且校正检验通过，才支持双任务共同通过。
"""
    (output_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def collect_results(checkpoint_dir: Path, subjects: list[SubjectInput]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for subject in subjects:
        path = checkpoint_file(checkpoint_dir, subject.subject_id)
        if not path.is_file():
            failures.append({"subject_id": subject.subject_id, "error_type": "MissingCheckpoint", "error_message": "No checkpoint"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "success":
            failures.append({
                "subject_id": subject.subject_id, "error_type": payload.get("error_type", "Unknown"),
                "error_message": payload.get("error_message", ""),
            })
            continue
        fold_rows.extend(payload["folds"])
        subject_rows.extend(payload["subjects"])
    return pd.DataFrame(subject_rows), pd.DataFrame(fold_rows), pd.DataFrame(failures)


def run_jobs(subjects: list[SubjectInput], args: argparse.Namespace) -> None:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending = [subject for subject in subjects if load_completed(args.checkpoint_dir, subject.subject_id) is None]
    print(f"subjects={len(subjects)} pending={len(pending)} workers={args.workers} permutations={args.permutations}", flush=True)
    if not pending:
        return
    # Threads avoid platform semaphore restrictions in the desktop sandbox.
    # SciPy's optimizer kernels and the Numba scorer release the GIL.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_subject, subject, args.permutations, args.seed): subject
            for subject in pending
        }
        completed = len(subjects) - len(pending)
        for future in as_completed(futures):
            subject = futures[future]
            try:
                payload = future.result()
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "status": "failed", "subject_id": subject.subject_id,
                    "error_type": type(exc).__name__, "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            write_checkpoint(checkpoint_file(args.checkpoint_dir, subject.subject_id), payload)
            completed += 1
            print(
                f"completed={completed}/{len(subjects)} subject={subject.subject_id} "
                f"status={payload.get('status')} runtime={payload.get('runtime_seconds', math.nan):.1f}s",
                flush=True,
            )


def final_qa(subjects: pd.DataFrame, folds: pd.DataFrame, failures: pd.DataFrame, expected_subjects: int) -> None:
    if not failures.empty:
        raise ValueError(f"Formal run has {len(failures)} failures")
    if len(subjects) != expected_subjects * 2:
        raise ValueError(f"Expected {expected_subjects * 2} participant-task rows, found {len(subjects)}")
    if len(folds) != expected_subjects * 2 * FOLDS:
        raise ValueError(f"Expected {expected_subjects * 2 * FOLDS} fold rows, found {len(folds)}")
    if not folds["fit_success"].astype(bool).all():
        raise ValueError("At least one cross-validation fit did not report success")
    numeric = [
        "real_score_per_observation", "shuffle_score_per_observation_mean",
        "shuffle_score_per_observation_sd", "delta_score_per_observation", "empirical_one_sided_p",
    ]
    if not np.isfinite(subjects[numeric].to_numpy(float)).all():
        raise ValueError("Non-finite participant result")
    if subjects.groupby("task")["subject_id"].nunique().to_dict() != {"T5": expected_subjects, "T6": expected_subjects}:
        raise ValueError("Task-specific participant counts do not match")


def main() -> None:
    args = parse_args()
    if args.permutations < 1:
        raise ValueError("permutations must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    subjects = read_subjects(args.manifest, args.max_subjects)
    if args.shard_count > 1:
        subjects = subjects[args.shard_index::args.shard_count]
    if not args.analysis_only:
        run_jobs(subjects, args)
    if args.checkpoint_only:
        return
    subject_frame, fold_frame, failures = collect_results(args.checkpoint_dir, subjects)
    final_qa(subject_frame, fold_frame, failures, len(subjects))
    summary = summarize(subject_frame, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_frame.sort_values(["task", "subject_id"]).to_csv(
        args.output_dir / "T5_T6_timeblock_permutation_subject_N118.csv", index=False, encoding="utf-8-sig",
    )
    fold_frame.sort_values(["task", "subject_id", "fold"]).to_csv(
        args.output_dir / "T5_T6_timeblock_permutation_folds_N118.csv", index=False, encoding="utf-8-sig",
    )
    summary.to_csv(args.output_dir / "T5_T6_timeblock_permutation_summary_N118.csv", index=False, encoding="utf-8-sig")
    save_figures(subject_frame, summary, args.output_dir)
    write_reports(subject_frame, fold_frame, summary, failures, args.output_dir, args.permutations)
    contract = {
        "task_id": TASK_ID, "status": "needs_review", "subjects": len(subjects),
        "folds": FOLDS, "block_ms": BLOCK_MS, "permutations": args.permutations,
        "seed": args.seed, "t5_resample_ratio": T5_RESAMPLE_RATIO,
        "inference_unit": "participant", "multiple_testing": "Holm across T5 and T6",
        "shared_null": "within-trial 250-ms complete-observation block permutation",
    }
    (args.output_dir / "analysis-contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if not args.keep_checkpoints:
        shutil.rmtree(args.checkpoint_dir)
    print(summary.to_string(index=False), flush=True)
    print(f"output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
