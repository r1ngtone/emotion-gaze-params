"""Portable entry for the 7% device-valid, 24-trial T5 recovery study.

Original seed rule, generator and optimizer are retained. The empirical truth
bank and the matching observed trial grids are supplied separately. The smoke
mode uses invented grids and parameter values.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fit.t5_memory.model import load_a5_dataframe, loglik_picture_only
from fit.t5_memory.pipeline import run_memory_pipeline_from_dataframes
from recovery.parameter_recovery_common import atomic_write_csv, finite_values, stable_seed

EMOTIONS = ("positive", "negative", "neutral", "complex")

RESAMPLE_RATIO = 0.07

RANDOM_SEED = 20260804

TRUTH_BANK_SIZE = 16

SIMULATIONS_PER_TRUTH = 60

MAIN_CONDITION = "main_24_trial"

PARAMETERS = tuple(
    f"{emotion}_{kind}"
    for emotion in ("positive", "negative", "neutral", "complex")
    for kind in ("tau", "sigma_s", "d_c")
)

STARTS = (
    {
        "name": "production_default",
        "theta": [0.5, 120.0, 80.0] * 4,
        "role": "primary",
    },
    {
        "name": "fit_api_default",
        "theta": [0.26, 27.0, 31.0, 0.18, 38.0, 32.0, 0.14, 34.0, 34.0, 0.15, 31.0, 28.0],
        "role": "diagnostic",
    },
    {
        "name": "fast_narrow",
        "theta": [0.05, 25.0, 25.0] * 4,
        "role": "diagnostic",
    },
    {
        "name": "staggered",
        "theta": [0.05, 25.0, 25.0, 0.1, 35.0, 35.0, 0.2, 50.0, 45.0, 0.4, 80.0, 60.0],
        "role": "diagnostic",
    },
    {
        "name": "reverse_staggered",
        "theta": [0.4, 80.0, 60.0, 0.2, 50.0, 45.0, 0.1, 35.0, 35.0, 0.05, 25.0, 25.0],
        "role": "diagnostic",
    },
)

def parameter_bounds() -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for parameter in PARAMETERS:
        if parameter == "complex_tau":
            bounds.append((0.0001, 10.0))
        elif parameter.endswith("_tau"):
            bounds.append((0.01, 10.0))
        elif parameter.endswith("_sigma_s"):
            bounds.append((5.0, 600.0))
        elif parameter.endswith("_d_c"):
            bounds.append((5.0, 400.0))
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
    return bounds


def build_grid(width: float, height: float, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(m) + 0.5) * (width / m)
    ys = (np.arange(n) + 0.5) * (height / n)
    return np.meshgrid(xs, ys, indexing="ij")

def generate_t5_dataset(
    source_data: dict,
    theta: np.ndarray,
    seed: int,
    m: int = 12,
    n: int = 9,
) -> dict:
    """Generate gaze using the same decay, mixture, observation and update order as the formal likelihood."""

    theta = np.asarray(theta, dtype=float)
    if theta.shape != (12,):
        raise ValueError("theta must have length 12")
    rng = np.random.default_rng(seed)
    x_centers, y_centers = build_grid(float(source_data["width"]), float(source_data["height"]), m, n)
    params = {
        emotion: {
            "tau": float(theta[index * 3]),
            "sigma_s": float(theta[index * 3 + 1]),
            "d_c": float(theta[index * 3 + 2]),
        }
        for index, emotion in enumerate(EMOTIONS)
    }
    generated_trials = []
    for source_trial in source_data["trials"]:
        emotion = str(source_trial["emotion"])
        if emotion not in params:
            raise ValueError(f"Unknown T5 emotion: {emotion}")
        times = np.asarray(source_trial["t"], dtype=float).copy()
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0):
            raise ValueError(f"Trial {source_trial['trial_number']} has an invalid time grid")
        emotion_params = params[emotion]
        attention = np.ones((m, n), dtype=float)
        x_values = np.empty(len(times), dtype=float)
        y_values = np.empty(len(times), dtype=float)
        last_time = None
        for index, current_time in enumerate(times):
            if last_time is not None:
                delta = max(0.0, float(current_time - last_time))
                attention *= math.exp(-delta / emotion_params["tau"])
            probabilities = attention.ravel() / attention.sum()
            center_index = int(rng.choice(m * n, p=probabilities))
            center_x = float(x_centers.ravel()[center_index])
            center_y = float(y_centers.ravel()[center_index])
            x_value = float(rng.normal(center_x, emotion_params["d_c"]))
            y_value = float(rng.normal(center_y, emotion_params["d_c"]))
            x_values[index] = x_value
            y_values[index] = y_value
            distance_squared = (x_centers - x_value) ** 2 + (y_centers - y_value) ** 2
            attention += np.exp(-distance_squared / (2.0 * emotion_params["sigma_s"] ** 2))
            last_time = float(current_time)
        generated_trials.append(
            {
                "trial_number": int(source_trial["trial_number"]),
                "emotion": emotion,
                "x": x_values,
                "y": y_values,
                "t": times,
                "n": len(times),
                "n_original": int(source_trial.get("n_original", len(times))),
            }
        )
    return {
        "width": float(source_data["width"]),
        "height": float(source_data["height"]),
        "emotions": list(source_data["emotions"]),
        "resample_ratio": float(source_data["resample_ratio"]),
        "trials": generated_trials,
    }

def fit_by_emotion_blocks(
    data: dict,
    start_theta: np.ndarray,
    bounds: list[tuple[float, float]],
    loglik_picture_only,
    m: int,
    n: int,
) -> dict:
    theta = np.asarray(start_theta, dtype=float).copy()
    messages = []
    successes = []
    started = time.perf_counter()
    for emotion_index, emotion in enumerate(EMOTIONS):
        block_slice = slice(emotion_index * 3, emotion_index * 3 + 3)
        emotion_data = {
            **data,
            "trials": [trial for trial in data["trials"] if trial["emotion"] == emotion],
        }

        def objective(block_theta: np.ndarray) -> float:
            full_theta = theta.copy()
            full_theta[block_slice] = block_theta
            return -float(loglik_picture_only(emotion_data, full_theta, M=m, N=n))

        result = minimize(
            objective,
            x0=theta[block_slice],
            method="L-BFGS-B",
            bounds=bounds[block_slice],
        )
        theta[block_slice] = result.x
        messages.append(str(result.message))
        successes.append(bool(result.success))
    log_likelihood = float(loglik_picture_only(data, theta, M=m, N=n))
    return {
        "theta": theta,
        "success": bool(all(successes) and math.isfinite(log_likelihood) and finite_values(theta)),
        "message": " | ".join(messages),
        "best_log_likelihood": log_likelihood,
        "runtime_seconds": time.perf_counter() - started,
    }


def load_source(raw_dir: Path) -> dict:
    eye = pd.read_csv(raw_dir / "formal_eyetrack.csv", encoding="utf-8-sig")
    records = pd.read_csv(raw_dir / "formal_records.csv", encoding="utf-8-sig")
    model_input = run_memory_pipeline_from_dataframes(eye, records)["A5"]
    source = load_a5_dataframe(model_input, timestamp_unit_scale=1e9, resample_ratio=RESAMPLE_RATIO)
    if any(sum(trial["emotion"] == emotion for trial in source["trials"]) != 6 for emotion in EMOTIONS):
        raise ValueError("Each T5 source must contain six usable trials per emotion")
    return source


def simulate_and_refit(truth_id: str, theta: np.ndarray, source: dict, repeat: int):
    seed = stable_seed(RANDOM_SEED, "T5", "7pct", truth_id, MAIN_CONDITION, repeat)
    run_id = f"t5_7pct_{truth_id}_{MAIN_CONDITION}_r{repeat:03d}"
    data = generate_t5_dataset(source, theta, seed)
    fits = []
    for start in STARTS:
        fit = fit_by_emotion_blocks(
            data, np.array(start["theta"]), parameter_bounds(), loglik_picture_only, 12, 9,
        )
        fitted = fit.pop("theta")
        fits.append({
            "run_id": run_id, "truth_id": truth_id, "sim": repeat,
            "start_name": start["name"],
            "fit_role": "formal" if start["role"] == "primary" else "diagnostic",
            **fit, **{f"theta_{i}": float(value) for i, value in enumerate(fitted)},
        })
    primary = fits[0]
    manifest = {
        "run_id": run_id, "truth_id": truth_id, "sim": repeat, "seed": seed,
        "n_trials": len(data["trials"]), "n_observations": sum(trial["n"] for trial in data["trials"]),
        "primary_finite": finite_values([primary["best_log_likelihood"], *(primary[f"theta_{i}"] for i in range(12))]),
        "primary_success": primary["success"],
    }
    return manifest, fits


def smoke_source() -> dict:
    trials = [
        {"trial_number": 6 * emotion_index + trial + 1, "emotion": emotion, "t": np.arange(20) * 0.075}
        for emotion_index, emotion in enumerate(EMOTIONS) for trial in range(6)
    ]
    return {"width": 960.0, "height": 720.0, "emotions": EMOTIONS, "resample_ratio": RESAMPLE_RATIO, "trials": trials}


def summarize_primary(sources: list, fits: list[dict]) -> pd.DataFrame:
    truths = pd.DataFrame(
        [(key, *theta) for key, theta, _ in sources],
        columns=["truth_id", *PARAMETERS],
    ).set_index("truth_id")
    primary = pd.DataFrame(row for row in fits if row["fit_role"] == "formal")
    summary = []
    for index, parameter in enumerate(PARAMETERS):
        true = truths[parameter]
        estimates = primary.groupby("truth_id")[f"theta_{index}"].median().reindex(true.index)
        errors = primary[f"theta_{index}"] - primary["truth_id"].map(true)
        span = true.max() - true.min()
        summary.append({
            "parameter": parameter,
            "n_datasets": len(primary),
            "truth_level_spearman_rho": float(true.rank().corr(estimates.rank())) if span > 0 else np.nan,
            "mae": float(errors.abs().mean()),
            "normalized_mae": float(errors.abs().mean() / span) if span > 0 else np.nan,
        })
    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--smoke", action="store_true")
    inputs.add_argument("--truth-bank", type=Path)
    parser.add_argument("--truth-id", action="append", help="Select a subset of rows for a local check")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repeats = args.repeats if args.repeats is not None else (1 if args.smoke else SIMULATIONS_PER_TRUTH)
    if not 1 <= repeats <= SIMULATIONS_PER_TRUTH:
        parser.error(f"repeats must be between 1 and {SIMULATIONS_PER_TRUTH}")
    if args.smoke:
        if args.truth_id:
            parser.error("--truth-id requires --truth-bank")
        sources = [("synthetic", np.array([0.4, 70.0, 55.0] * 4), smoke_source())]
    else:
        bank = pd.read_csv(args.truth_bank, encoding="utf-8-sig", dtype={"truth_id": str})
        missing = {"truth_id", "raw_dir", *PARAMETERS} - set(bank.columns)
        if missing:
            parser.error(f"missing truth-bank columns: {sorted(missing)}")
        if len(bank) != TRUTH_BANK_SIZE or bank["truth_id"].nunique() != TRUTH_BANK_SIZE:
            parser.error(f"the historical bank must contain {TRUTH_BANK_SIZE} unique truth IDs")
        if args.truth_id:
            if not set(args.truth_id) <= set(bank["truth_id"]):
                parser.error("unknown truth ID")
            bank = bank.loc[bank["truth_id"].isin(args.truth_id)]
        sources = []
        for row in bank.to_dict("records"):
            theta = np.array([row[name] for name in PARAMETERS], dtype=float)
            if not all(lo <= value <= hi for value, (lo, hi) in zip(theta, parameter_bounds())):
                parser.error(f"non-finite or out-of-bounds truth: {row['truth_id']}")
            raw_dir = args.truth_bank.parent / str(row["raw_dir"])
            sources.append((row["truth_id"], theta, load_source(raw_dir)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")
    manifests, fits = [], []
    for truth_id, theta, source in sources:
        for repeat in range(1, repeats + 1):
            manifest, starts = simulate_and_refit(truth_id, theta, source, repeat)
            manifests.append(manifest)
            fits.extend(starts)
        atomic_write_csv(pd.DataFrame(manifests), args.output_dir / "simulation_manifest.csv")
        atomic_write_csv(pd.DataFrame(fits), args.output_dir / "recovery_runs.csv")
    if not all(row["primary_finite"] for row in manifests):
        raise RuntimeError("At least one primary T5 estimate was non-finite")
    if len(sources) > 1:
        atomic_write_csv(summarize_primary(sources, fits), args.output_dir / "parameter-summary.csv")
    print(f"T5: {len(manifests)} finite primary estimates, {len(fits)} total fits")


if __name__ == "__main__":
    main()
