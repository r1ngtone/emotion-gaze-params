#!/usr/bin/env python3
"""Run deterministic T5 parameter-recovery simulation and fitting.

T60c-a: smoke mode — validates generator, fitting pipeline, and truth-near
        diagnostics on a single truth row (3 repeats).
T60c-b: full mode  — runs all 16 truths × 60 datasets × 5 formal starts,
        plus sensitivity conditions and programmatic parameter-level decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from parameter_recovery_common import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    SCRIPT_DIR,
    atomic_write_csv,
    atomic_write_text,
    boundary_label,
    build_contract,
    finite_values,
    load_config,
    parameter_kind,
    stable_seed,
)


T5_OUTPUT = SCRIPT_DIR / "outputs" / "T5"
SMOKE_OUTPUT = T5_OUTPUT / "smoke"
EMOTIONS = ("positive", "negative", "neutral", "complex")
SMOKE_REPEATS = 3
TRUTH_NEAR_FRACTION = 0.01
TRUTH_NEAR_MAX_SCALED_ERROR = 0.50
TRUTH_NEAR_MEDIAN_SCALED_ERROR = 0.20

# --- T60c-b full-mode constants ---
FULL_OUTPUT = T5_OUTPUT / "full"
SIMULATIONS_PER_TRUTH = 60
SENSITIVITY_CONDITIONS = {
    "main_24_trial": {"n_trials": 24, "label": "main"},
    "half_12_trial": {"n_trials": 12, "label": "half_trial"},
    "late_8_trial": {"n_trials": 8, "label": "late_only"},
}
RECOVERY_THRESHOLDS = {
    "strong": {"rho": 0.70, "bias_pct": 15.0, "coverage": 0.80},
    "cautious": {"rho": 0.40, "bias_pct": 30.0, "coverage": 0.50},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def t5_module_imports() -> tuple:
    t5_root = REPO_ROOT / "single_pic" / "单图" / "t5_divide" / "src"
    if str(t5_root) not in sys.path:
        sys.path.insert(0, str(t5_root))
    from t5_memory.model import (
        compute_interpretable_metrics,
        load_a5_dataframe,
        loglik_picture_only,
    )
    from t5_memory.pipeline import run_memory_pipeline

    return compute_interpretable_metrics, load_a5_dataframe, loglik_picture_only, run_memory_pipeline


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_contract(config: dict, config_path: Path) -> dict:
    base = build_contract(config, config_path)
    payload = {
        "base_contract_hash": base["contract_hash"],
        "runner_sha256": sha256_file(Path(__file__)),
        "mode": "smoke",
        "smoke_repeats": SMOKE_REPEATS,
        "truth_near_fraction": TRUTH_NEAR_FRACTION,
        "truth_near_max_scaled_error": TRUTH_NEAR_MAX_SCALED_ERROR,
        "truth_near_median_scaled_error": TRUTH_NEAR_MEDIAN_SCALED_ERROR,
        "random_seed": int(config["random_seed"]),
        "starts": config["t5"]["starts"],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["contract_hash"] = hashlib.sha256(canonical).hexdigest()
    payload["base_contract"] = base
    return payload


def theta_from_row(row, parameter_names: list[str]) -> np.ndarray:
    if isinstance(row, dict):
        theta = np.array([float(row[name]) for name in parameter_names], dtype=float)
    else:
        theta = row[parameter_names].to_numpy(dtype=float)
    if theta.shape != (12,) or not finite_values(theta):
        raise ValueError("T5 truth row must contain 12 finite free parameters")
    return theta


def parameter_bounds(config: dict) -> list[tuple[float, float]]:
    return [
        tuple(float(value) for value in config["t5"]["bounds_by_kind"][parameter_kind(name)])
        for name in config["t5"]["parameter_names"]
    ]


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


def datasets_equal(first: dict, second: dict) -> bool:
    if len(first["trials"]) != len(second["trials"]):
        return False
    for left, right in zip(first["trials"], second["trials"]):
        if left["trial_number"] != right["trial_number"] or left["emotion"] != right["emotion"]:
            return False
        for key in ("x", "y", "t"):
            if not np.array_equal(left[key], right[key]):
                return False
    return True


def sequence_matches_source(generated: dict, source: dict) -> bool:
    if len(generated["trials"]) != len(source["trials"]):
        return False
    for generated_trial, source_trial in zip(generated["trials"], source["trials"]):
        if (
            generated_trial["trial_number"] != source_trial["trial_number"]
            or generated_trial["emotion"] != source_trial["emotion"]
            or not np.array_equal(generated_trial["t"], source_trial["t"])
        ):
            return False
    return True


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


def candidate_scales(candidate_bank: pd.DataFrame, parameter_names: list[str]) -> np.ndarray:
    values = candidate_bank[parameter_names].to_numpy(dtype=float)
    p05, p95 = np.nanpercentile(values, [5, 95], axis=0)
    scales = p95 - p05
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Candidate P95-P05 parameter scales must be finite and positive")
    return scales


def truth_near_start(
    truth: np.ndarray,
    scales: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    directions = np.where(np.arange(len(truth)) % 2 == 0, 1.0, -1.0)
    values = truth + directions * TRUTH_NEAR_FRACTION * scales
    lower = np.asarray([item[0] for item in bounds])
    upper = np.asarray([item[1] for item in bounds])
    epsilon = np.maximum((upper - lower) * 1e-8, 1e-12)
    return np.clip(values, lower + epsilon, upper - epsilon)


def derived_metrics_flat(data: dict, theta: np.ndarray, compute_interpretable_metrics, m: int, n: int, r_focus: float) -> dict:
    metrics = compute_interpretable_metrics(data, theta, M=m, N=n, r_focus=r_focus)
    return {
        f"{emotion}_{metric}": float(metrics[emotion][metric])
        for emotion in EMOTIONS
        for metric in ("T_half", "H_bar_norm", "I_focus")
    }


def stable_run_id(truth_id: str, repeat: int) -> str:
    return f"t5_smoke_{truth_id}_main_r{repeat:02d}"


def load_source_data(truth_row: pd.Series, config: dict) -> dict:
    _, load_a5_dataframe, _, run_memory_pipeline = t5_module_imports()
    raw_dir = Path(str(truth_row["raw_dir"]))
    pipeline_data = run_memory_pipeline(
        str(raw_dir / "formal_eyetrack.csv"),
        str(raw_dir / "formal_records.csv"),
    )
    return load_a5_dataframe(
        pipeline_data["A5"],
        timestamp_unit_scale=float(config["t5"]["timestamp_unit_scale"]),
        width=float(config["t5"]["width"]),
        height=float(config["t5"]["height"]),
        resample_ratio=float(config["t5"]["resample_ratio"]),
    )


def validate_resume(frame: pd.DataFrame, contract_hash: str, path: Path) -> pd.DataFrame:
    if frame.empty:
        return frame
    hashes = set(frame["contract_hash"].astype(str))
    if hashes != {contract_hash}:
        raise ValueError(f"Resume contract mismatch in {path}: {hashes}")
    return frame


def run_smoke(config: dict, contract: dict, resume: bool) -> dict:
    SMOKE_OUTPUT.mkdir(parents=True, exist_ok=True)
    truth_bank = pd.read_csv(T5_OUTPUT / "truth_parameter_bank.csv", dtype={"subject_id": str})
    candidate_bank = pd.read_csv(T5_OUTPUT / "candidate_parameter_bank_N118.csv", dtype={"subject_id": str})
    if len(truth_bank) != 16 or truth_bank["truth_id"].nunique() != 16:
        raise ValueError("T60c-a requires exactly 16 unique T5 truth rows")
    truth_row = truth_bank.sort_values("selection_order").iloc[0]
    parameter_names = list(config["t5"]["parameter_names"])
    truth_theta = theta_from_row(truth_row, parameter_names)
    scales = candidate_scales(candidate_bank, parameter_names)
    bounds = parameter_bounds(config)
    source_data = load_source_data(truth_row, config)
    if len(source_data["trials"]) != int(config["t5"]["expected_trials"]):
        raise ValueError(f"Smoke truth source requires 24 trials, found {len(source_data['trials'])}")

    manifest_path = SMOKE_OUTPUT / "simulation_manifest.csv"
    runs_path = SMOKE_OUTPUT / "recovery_runs.csv"
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str}) if resume and manifest_path.is_file() else pd.DataFrame()
    runs = pd.read_csv(runs_path, dtype={"subject_id": str}) if resume and runs_path.is_file() else pd.DataFrame()
    manifest = validate_resume(manifest, contract["contract_hash"], manifest_path)
    runs = validate_resume(runs, contract["contract_hash"], runs_path)
    completed = set(manifest.loc[manifest["status"].eq("complete"), "run_id"]) if not manifest.empty else set()
    manifest_rows = manifest.to_dict("records")
    recovery_rows = runs.to_dict("records")
    _, _, loglik_picture_only, _ = t5_module_imports()
    compute_interpretable_metrics = t5_module_imports()[0]
    m = int(config["t5"]["grid_m"])
    n = int(config["t5"]["grid_n"])
    r_focus = float(config["t5"]["r_focus"])

    for repeat in range(1, SMOKE_REPEATS + 1):
        run_id = stable_run_id(str(truth_row["truth_id"]), repeat)
        if run_id in completed:
            continue
        seed = stable_seed(int(config["random_seed"]), "T5", "smoke", truth_row["truth_id"], repeat)
        generated = generate_t5_dataset(source_data, truth_theta, seed, m=m, n=n)
        repeated = generate_t5_dataset(source_data, truth_theta, seed, m=m, n=n)
        reproducible = datasets_equal(generated, repeated)
        sequence_valid = sequence_matches_source(generated, source_data)
        all_points = np.concatenate(
            [np.column_stack([trial["x"], trial["y"]]) for trial in generated["trials"]],
            axis=0,
        )
        finite_support = bool(np.isfinite(all_points).all())
        outside_nominal = int(
            np.sum(
                (all_points[:, 0] < 0)
                | (all_points[:, 0] > float(config["t5"]["width"]))
                | (all_points[:, 1] < 0)
                | (all_points[:, 1] > float(config["t5"]["height"]))
            )
        )
        run_attempts: list[dict] = []
        for start in config["t5"]["starts"]:
            fit = fit_by_emotion_blocks(
                generated,
                np.asarray(start["theta"], dtype=float),
                bounds,
                loglik_picture_only,
                m,
                n,
            )
            row = {
                "run_id": run_id,
                "truth_id": truth_row["truth_id"],
                "subject_id": truth_row["subject_id"],
                "repeat": repeat,
                "condition": "main_24_trial",
                "start_name": start["name"],
                "fit_role": "formal",
                "success": fit["success"],
                "message": fit["message"],
                "best_log_likelihood": fit["best_log_likelihood"],
                "runtime_seconds": fit["runtime_seconds"],
                "contract_hash": contract["contract_hash"],
                "is_best_formal": False,
            }
            row.update({name: float(value) for name, value in zip(parameter_names, fit["theta"])})
            row.update(derived_metrics_flat(generated, fit["theta"], compute_interpretable_metrics, m, n, r_focus))
            run_attempts.append(row)

        diagnostic_fit = fit_by_emotion_blocks(
            generated,
            truth_near_start(truth_theta, scales, bounds),
            bounds,
            loglik_picture_only,
            m,
            n,
        )
        diagnostic_row = {
            "run_id": run_id,
            "truth_id": truth_row["truth_id"],
            "subject_id": truth_row["subject_id"],
            "repeat": repeat,
            "condition": "main_24_trial",
            "start_name": "truth_near_diagnostic",
            "fit_role": "diagnostic",
            "success": diagnostic_fit["success"],
            "message": diagnostic_fit["message"],
            "best_log_likelihood": diagnostic_fit["best_log_likelihood"],
            "runtime_seconds": diagnostic_fit["runtime_seconds"],
            "contract_hash": contract["contract_hash"],
            "is_best_formal": False,
        }
        diagnostic_row.update(
            {name: float(value) for name, value in zip(parameter_names, diagnostic_fit["theta"])}
        )
        diagnostic_row.update(
            derived_metrics_flat(generated, diagnostic_fit["theta"], compute_interpretable_metrics, m, n, r_focus)
        )
        run_attempts.append(diagnostic_row)
        valid_formal = [row for row in run_attempts if row["fit_role"] == "formal" and row["success"]]
        if valid_formal:
            max(valid_formal, key=lambda row: row["best_log_likelihood"])["is_best_formal"] = True
        recovery_rows.extend(run_attempts)
        truth_loglik = float(loglik_picture_only(generated, truth_theta, M=m, N=n))
        manifest_rows.append(
            {
                "run_id": run_id,
                "truth_id": truth_row["truth_id"],
                "subject_id": truth_row["subject_id"],
                "repeat": repeat,
                "condition": "main_24_trial",
                "seed": seed,
                "n_trials": len(generated["trials"]),
                "n_observations": int(sum(trial["n"] for trial in generated["trials"])),
                "support": "R2",
                "outside_nominal_960x720": outside_nominal,
                "finite_support": finite_support,
                "seed_reproducible": reproducible,
                "sequence_matches_source": sequence_valid,
                "truth_log_likelihood": truth_loglik,
                "formal_success_count": len(valid_formal),
                "diagnostic_success": diagnostic_fit["success"],
                "status": "complete",
                "contract_hash": contract["contract_hash"],
            }
        )
        atomic_write_csv(pd.DataFrame(manifest_rows).sort_values("run_id"), manifest_path)
        atomic_write_csv(
            pd.DataFrame(recovery_rows).sort_values(["run_id", "fit_role", "start_name"]),
            runs_path,
        )
        print(
            f"[T5 smoke] {run_id} formal={len(valid_formal)}/5 diagnostic={diagnostic_fit['success']}",
            flush=True,
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("run_id").reset_index(drop=True)
    runs = pd.DataFrame(recovery_rows).sort_values(["run_id", "fit_role", "start_name"]).reset_index(drop=True)
    if manifest["run_id"].duplicated().any() or runs.duplicated(["run_id", "start_name"]).any():
        raise ValueError("Duplicate smoke run ID or run/start pair")
    diagnostic = runs[runs["fit_role"].eq("diagnostic")].copy()
    error_rows = []
    for _, row in diagnostic.iterrows():
        errors = np.abs(row[parameter_names].to_numpy(dtype=float) - truth_theta) / scales
        error_rows.append(
            {
                "run_id": row["run_id"],
                "max_scaled_error": float(np.max(errors)),
                "median_scaled_error": float(np.median(errors)),
                "truth_near_pass": bool(
                    row["success"]
                    and np.max(errors) <= TRUTH_NEAR_MAX_SCALED_ERROR
                    and np.median(errors) <= TRUTH_NEAR_MEDIAN_SCALED_ERROR
                ),
                **{
                    f"{name}_scaled_error": float(error)
                    for name, error in zip(parameter_names, errors)
                },
                "contract_hash": contract["contract_hash"],
            }
        )
    round_trip = pd.DataFrame(error_rows)
    atomic_write_csv(round_trip, SMOKE_OUTPUT / "round_trip_summary.csv")
    expected_runs = SMOKE_REPEATS
    expected_fits = SMOKE_REPEATS * (len(config["t5"]["starts"]) + 1)
    generation_pass = bool(
        len(manifest) == expected_runs
        and manifest["seed_reproducible"].all()
        and manifest["sequence_matches_source"].all()
        and manifest["finite_support"].all()
        and manifest["n_trials"].eq(int(config["t5"]["expected_trials"])).all()
    )
    fit_pass = bool(
        len(runs) == expected_fits
        and runs[runs["fit_role"].eq("formal")]["success"].all()
        and runs[runs["fit_role"].eq("formal")].groupby("run_id")["is_best_formal"].sum().eq(1).all()
    )
    truth_near_pass = bool(len(round_trip) == expected_runs and round_trip["truth_near_pass"].all())
    smoke_pass = bool(generation_pass and fit_pass and truth_near_pass)
    decisions = pd.DataFrame(
        [
            {
                "check_id": "generation_contract",
                "decision": "PASS" if generation_pass else "BLOCKED",
                "rule": "3 unique runs; fixed-seed exact reproduction; 24 source-aligned trials; finite R2 samples",
                "observed_value": (
                    f"runs={len(manifest)}/{expected_runs};"
                    f"reproducible={int(manifest['seed_reproducible'].sum())};"
                    f"sequence_valid={int(manifest['sequence_matches_source'].sum())}"
                ),
                "programmatic": True,
                "contract_hash": contract["contract_hash"],
            },
            {
                "check_id": "formal_round_trip_fits",
                "decision": "PASS" if fit_pass else "BLOCKED",
                "rule": "18 fit rows total; all 15 formal fits successful; one formal best per run",
                "observed_value": (
                    f"fit_rows={len(runs)}/{expected_fits};"
                    f"formal_success={int(runs[runs['fit_role'].eq('formal')]['success'].sum())}/15"
                ),
                "programmatic": True,
                "contract_hash": contract["contract_hash"],
            },
            {
                "check_id": "truth_near_diagnostic",
                "decision": "PASS" if truth_near_pass else "BLOCKED",
                "rule": "each diagnostic max scaled error<=0.50 and median scaled error<=0.20",
                "observed_value": (
                    f"pass={int(round_trip['truth_near_pass'].sum())}/{expected_runs};"
                    f"max_scaled_error={round_trip['max_scaled_error'].max():.6g};"
                    f"max_median_scaled_error={round_trip['median_scaled_error'].max():.6g}"
                ),
                "programmatic": True,
                "contract_hash": contract["contract_hash"],
            },
            {
                "check_id": "t60c_b_admission",
                "decision": "READY" if smoke_pass else "BLOCKED",
                "rule": "generation_contract=PASS AND formal_round_trip_fits=PASS AND truth_near_diagnostic=PASS",
                "observed_value": (
                    f"generation={generation_pass};fits={fit_pass};truth_near={truth_near_pass}"
                ),
                "programmatic": True,
                "contract_hash": contract["contract_hash"],
            },
        ]
    )
    atomic_write_csv(decisions, SMOKE_OUTPUT / "smoke_decisions.csv")
    atomic_write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True),
        SMOKE_OUTPUT / "run_contract.json",
    )
    report = [
        "# T60c-a T5 Parameter Recovery Smoke",
        "",
        f"- Truth: {truth_row['truth_id']} / subject {truth_row['subject_id']}.",
        f"- Simulated datasets: {len(manifest)}/{expected_runs}.",
        f"- Formal fits: {int(runs[runs['fit_role'].eq('formal')]['success'].sum())}/15.",
        f"- Truth-near diagnostics passed: {int(round_trip['truth_near_pass'].sum())}/{expected_runs}.",
        f"- Programmatic admission to T60c-b: {'READY' if smoke_pass else 'BLOCKED'}.",
        "- Samples use the model's R2 support; outside nominal 960x720 samples are reported, not clipped.",
        "- No full recovery simulation was started.",
    ]
    atomic_write_text("\n".join(report) + "\n", SMOKE_OUTPUT / "analysis-report.md")
    return {
        "truth_id": str(truth_row["truth_id"]),
        "subject_id": str(truth_row["subject_id"]),
        "simulation_rows": len(manifest),
        "recovery_rows": len(runs),
        "formal_success": int(runs[runs["fit_role"].eq("formal")]["success"].sum()),
        "truth_near_pass": int(round_trip["truth_near_pass"].sum()),
        "admission": "READY" if smoke_pass else "BLOCKED",
    }


def full_contract(config: dict, config_path: Path) -> dict:
    base = build_contract(config, config_path)
    payload = {
        "base_contract_hash": base["contract_hash"],
        "runner_sha256": sha256_file(Path(__file__)),
        "mode": "full",
        "simulations_per_truth": SIMULATIONS_PER_TRUTH,
        "conditions": SENSITIVITY_CONDITIONS,
        "thresholds": RECOVERY_THRESHOLDS,
        "random_seed": int(config["random_seed"]),
        "starts": config["t5"]["starts"],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["contract_hash"] = hashlib.sha256(canonical).hexdigest()
    payload["base_contract"] = base
    return payload


def generate_sensitivity_datasets(
    source_data: dict,
    theta: np.ndarray,
    seed: int,
    condition: str,
    m: int = 12,
    n: int = 9,
) -> dict:
    """Generate a dataset for a specific sensitivity condition."""
    spec = SENSITIVITY_CONDITIONS[condition]
    n_trials_target = spec["n_trials"]
    if n_trials_target > len(source_data["trials"]):
        raise ValueError(f"Condition {condition} wants {n_trials_target} trials, source has {len(source_data['trials'])}")
    subset_source = {**source_data, "trials": source_data["trials"][:n_trials_target]}
    return generate_t5_dataset(subset_source, theta, seed, m=m, n=n)


def fit_single_dataset(
    data: dict,
    starts: list[dict],
    bounds: list[tuple[float, float]],
    truth_theta: np.ndarray,
    scales: np.ndarray,
    loglik_picture_only,
    compute_interpretable_metrics,
    m: int,
    n: int,
    r_focus: float,
) -> list[dict]:
    """Fit all starts + truth-near diagnostic for one dataset."""
    rows = []
    for start in starts:
        fit = fit_by_emotion_blocks(
            data,
            np.asarray(start["theta"], dtype=float),
            bounds,
            loglik_picture_only,
            m,
            n,
        )
        row = {
            "start_name": start["name"],
            "fit_role": "formal",
            "success": fit["success"],
            "message": fit["message"],
            "best_log_likelihood": fit["best_log_likelihood"],
            "runtime_seconds": fit["runtime_seconds"],
        }
        row.update({name: float(value) for name, value in zip(
            [f"theta_{i}" for i in range(12)], fit["theta"])})
        row.update(derived_metrics_flat(data, fit["theta"], compute_interpretable_metrics, m, n, r_focus))
        rows.append(row)

    diagnostic_fit = fit_by_emotion_blocks(
        data,
        truth_near_start(truth_theta, scales, bounds),
        bounds,
        loglik_picture_only,
        m,
        n,
    )
    diag_row = {
        "start_name": "truth_near_diagnostic",
        "fit_role": "diagnostic",
        "success": diagnostic_fit["success"],
        "message": diagnostic_fit["message"],
        "best_log_likelihood": diagnostic_fit["best_log_likelihood"],
        "runtime_seconds": diagnostic_fit["runtime_seconds"],
    }
    diag_row.update({name: float(value) for name, value in zip(
        [f"theta_{i}" for i in range(12)], diagnostic_fit["theta"])})
    diag_row.update(derived_metrics_flat(data, diagnostic_fit["theta"], compute_interpretable_metrics, m, n, r_focus))
    rows.append(diag_row)

    valid_formal = [r for r in rows if r["fit_role"] == "formal" and r["success"]]
    if valid_formal:
        max(valid_formal, key=lambda r: r["best_log_likelihood"])["is_best_formal"] = True
    return rows


def compute_recovery_metrics(
    truth_values: np.ndarray,
    recovered_values: np.ndarray,
    scales: np.ndarray,
) -> dict:
    """Compute per-parameter recovery metrics."""
    finite_mask = np.isfinite(recovered_values)
    if finite_mask.sum() < 3:
        return {"n_valid": int(finite_mask.sum()), "rho": np.nan, "bias": np.nan,
                "bias_pct": np.nan, "rmse": np.nan, "scaled_rmse": np.nan,
                "coverage_50": np.nan, "coverage_95": np.nan}
    truth_finite = truth_values[finite_mask]
    recovered_finite = recovered_values[finite_mask]
    errors = recovered_finite - truth_finite
    bias = float(np.mean(errors))
    bias_pct = float(np.mean(np.abs(errors) / np.maximum(np.abs(truth_finite), 1e-8)) * 100.0)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_abs_error = float(np.median(np.abs(errors)))
    scaled_error = np.abs(errors) / np.maximum(scales[finite_mask], 1e-8)
    coverage_50 = float(np.mean(scaled_error <= 0.6745))  # 0.6745 ≈ IQR half-width
    coverage_95 = float(np.mean(scaled_error <= 1.96))
    if len(recovered_finite) >= 3:
        rho = float(np.corrcoef(truth_finite, recovered_finite)[0, 1])
    else:
        rho = np.nan
    return {
        "n_valid": int(finite_mask.sum()),
        "rho": rho,
        "bias": bias,
        "bias_pct": bias_pct,
        "rmse": rmse,
        "median_abs_error": median_abs_error,
        "scaled_rmse": float(np.sqrt(np.mean(scaled_error ** 2))),
        "coverage_50": coverage_50,
        "coverage_95": coverage_95,
    }


def parameter_support_level(metrics: dict, thresholds: dict) -> str:
    """Classify parameter support level from recovery metrics."""
    if metrics["n_valid"] < 3:
        return "insufficient"
    rho = metrics["rho"]
    bias_pct = metrics["bias_pct"]
    cov50 = metrics["coverage_50"]
    if not np.isfinite(rho):
        return "insufficient"
    strong = thresholds["strong"]
    cautious = thresholds["cautious"]
    if rho >= strong["rho"] and bias_pct <= strong["bias_pct"] and cov50 >= strong["coverage"]:
        return "strong"
    if rho >= cautious["rho"] and bias_pct <= cautious["bias_pct"] and cov50 >= cautious["coverage"]:
        return "cautious"
    return "not_supported"


def run_full(config: dict, contract: dict, resume: bool) -> dict:
    FULL_OUTPUT.mkdir(parents=True, exist_ok=True)
    truth_bank = pd.read_csv(T5_OUTPUT / "truth_parameter_bank.csv", dtype={"subject_id": str})
    candidate_bank = pd.read_csv(T5_OUTPUT / "candidate_parameter_bank_N118.csv", dtype={"subject_id": str})
    if len(truth_bank) < 1:
        raise ValueError(f"Need at least 1 truth row, found {len(truth_bank)}")

    parameter_names = list(config["t5"]["parameter_names"])
    parameter_names_theta = [f"theta_{i}" for i in range(12)]
    param_name_map = dict(zip(parameter_names_theta, parameter_names))
    scales = candidate_scales(candidate_bank, parameter_names)
    bounds = parameter_bounds(config)
    starts = config["t5"]["starts"]
    m = int(config["t5"]["grid_m"])
    n = int(config["t5"]["grid_n"])
    r_focus = float(config["t5"]["r_focus"])

    manifest_path = FULL_OUTPUT / "simulation_manifest.csv"
    runs_path = FULL_OUTPUT / "recovery_runs.csv"
    manifest = pd.read_csv(manifest_path, dtype={"truth_id": str}) if resume and manifest_path.is_file() else pd.DataFrame()
    runs = pd.read_csv(runs_path, dtype={"run_id": str}) if resume and runs_path.is_file() else pd.DataFrame()
    manifest = validate_resume(manifest, contract["contract_hash"], manifest_path) if not manifest.empty else manifest
    runs = validate_resume(runs, contract["contract_hash"], runs_path) if not runs.empty else runs
    completed = set(manifest.loc[manifest["status"].eq("complete"), "run_id"]) if not manifest.empty else set()
    manifest_rows = manifest.to_dict("records") if not manifest.empty else []
    recovery_rows = runs.to_dict("records") if not runs.empty else []

    _, _, loglik_picture_only, _ = t5_module_imports()
    compute_interpretable_metrics = t5_module_imports()[0]
    truth_rows = truth_bank.sort_values("selection_order").to_dict("records")
    total_expected = len(truth_rows) * len(SENSITIVITY_CONDITIONS) * SIMULATIONS_PER_TRUTH
    completed_count = len(completed)

    if completed_count >= total_expected:
        print(f"[T5 full] All {total_expected} runs already complete; skipping to metric computation.")
    else:
        print(f"[T5 full] Resuming with {completed_count}/{total_expected} completed; "
              f"{total_expected - completed_count} remaining.", flush=True)

    for truth_idx, truth_row in enumerate(truth_rows):
        truth_id = str(truth_row["truth_id"])
        truth_theta = theta_from_row(truth_row, parameter_names)
        try:
            source_data = load_source_data(truth_row, config)
        except Exception as exc:
            print(f"[T5 full] SKIP {truth_id}: cannot load source data — {exc}", flush=True)
            continue
        if len(source_data["trials"]) != int(config["t5"]["expected_trials"]):
            print(f"[T5 full] SKIP {truth_id}: expected 24 trials, found {len(source_data['trials'])}", flush=True)
            continue

        for condition_name, condition_spec in SENSITIVITY_CONDITIONS.items():
            n_trials_target = condition_spec["n_trials"]
            for sim in range(1, SIMULATIONS_PER_TRUTH + 1):
                run_id = f"t5_full_{truth_id}_{condition_name}_r{sim:03d}"
                if run_id in completed:
                    continue
                seed = stable_seed(int(config["random_seed"]), "T5", "full", truth_id, condition_name, sim)
                try:
                    generated = generate_sensitivity_datasets(
                        source_data, truth_theta, seed, condition_name, m=m, n=n)
                except Exception as exc:
                    print(f"[T5 full] GEN FAIL {run_id}: {exc}", flush=True)
                    manifest_rows.append({
                        "run_id": run_id, "truth_id": truth_id,
                        "subject_id": str(truth_row["subject_id"]),
                        "condition": condition_name, "sim": sim, "seed": seed,
                        "n_trials": n_trials_target, "status": "generation_failed",
                        "contract_hash": contract["contract_hash"],
                    })
                    atomic_write_csv(pd.DataFrame(manifest_rows).sort_values("run_id"), manifest_path)
                    continue

                all_points = np.concatenate(
                    [np.column_stack([t["x"], t["y"]]) for t in generated["trials"]], axis=0)
                finite_support = bool(np.isfinite(all_points).all())
                if not finite_support:
                    print(f"[T5 full] FINITE FAIL {run_id}", flush=True)
                    manifest_rows.append({
                        "run_id": run_id, "truth_id": truth_id,
                        "subject_id": str(truth_row["subject_id"]),
                        "condition": condition_name, "sim": sim, "seed": seed,
                        "n_trials": n_trials_target, "status": "finite_check_failed",
                        "contract_hash": contract["contract_hash"],
                    })
                    atomic_write_csv(pd.DataFrame(manifest_rows).sort_values("run_id"), manifest_path)
                    continue

                fit_rows = fit_single_dataset(
                    generated, starts, bounds, truth_theta, scales,
                    loglik_picture_only, compute_interpretable_metrics, m, n, r_focus)
                truth_loglik = float(loglik_picture_only(generated, truth_theta, M=m, N=n))
                valid_formal = [r for r in fit_rows if r["fit_role"] == "formal" and r["success"]]
                diagnostic = [r for r in fit_rows if r["fit_role"] == "diagnostic"]

                for fr in fit_rows:
                    fr["run_id"] = run_id
                    fr["truth_id"] = truth_id
                    fr["subject_id"] = str(truth_row["subject_id"])
                    fr["condition"] = condition_name
                    fr["sim"] = sim
                    fr["contract_hash"] = contract["contract_hash"]
                    fr.setdefault("is_best_formal", False)
                recovery_rows.extend(fit_rows)

                manifest_rows.append({
                    "run_id": run_id, "truth_id": truth_id,
                    "subject_id": str(truth_row["subject_id"]),
                    "condition": condition_name, "sim": sim, "seed": seed,
                    "n_trials": n_trials_target,
                    "n_observations": int(sum(t["n"] for t in generated["trials"])),
                    "finite_support": finite_support,
                    "truth_log_likelihood": truth_loglik,
                    "formal_success_count": len(valid_formal),
                    "diagnostic_success": diagnostic[0]["success"] if diagnostic else False,
                    "status": "complete",
                    "contract_hash": contract["contract_hash"],
                })

                if len(manifest_rows) % 50 == 0 or len(manifest_rows) == total_expected:
                    atomic_write_csv(
                        pd.DataFrame(manifest_rows).sort_values("run_id"), manifest_path)
                    atomic_write_csv(
                        pd.DataFrame(recovery_rows).sort_values(["run_id", "fit_role", "start_name"]),
                        runs_path)
                    print(f"[T5 full] progress {len(manifest_rows)}/{total_expected} rows; "
                          f"last={run_id} formal={len(valid_formal)}/5", flush=True)

    atomic_write_csv(pd.DataFrame(manifest_rows).sort_values("run_id"), manifest_path)
    atomic_write_csv(pd.DataFrame(recovery_rows).sort_values(["run_id", "fit_role", "start_name"]), runs_path)

    # --- Recovery metrics ---
    manifest_final = pd.read_csv(manifest_path, dtype={"truth_id": str})
    runs_final = pd.read_csv(runs_path, dtype={"run_id": str})
    main_runs = runs_final[runs_final["condition"].eq("main_24_trial")]
    formal_best = main_runs[(main_runs["fit_role"].eq("formal")) & (main_runs["is_best_formal"].eq(True))].copy()

    parameter_metrics_rows = []
    for theta_col, param_name in param_name_map.items():
        truth_values_list = []
        recovered_values_list = []
        for _, run_row in formal_best.iterrows():
            truth_row_match = truth_bank[truth_bank["truth_id"].eq(str(run_row["truth_id"]))]
            if truth_row_match.empty:
                continue
            truth_val = float(truth_row_match.iloc[0][param_name])
            rec_val = float(run_row[theta_col])
            if np.isfinite(truth_val) and np.isfinite(rec_val):
                truth_values_list.append(truth_val)
                recovered_values_list.append(rec_val)
        if len(truth_values_list) < 3:
            parameter_metrics_rows.append({"parameter": param_name, "n_pairs": len(truth_values_list),
                                            "support": "insufficient"})
            continue
        truth_arr = np.array(truth_values_list)
        recovered_arr = np.array(recovered_values_list)
        param_idx = parameter_names.index(param_name)
        param_scale = np.array([scales[param_idx]] * len(truth_arr))
        metrics = compute_recovery_metrics(truth_arr, recovered_arr, param_scale)
        support = parameter_support_level(metrics, RECOVERY_THRESHOLDS)
        metrics["parameter"] = param_name
        metrics["support"] = support
        parameter_metrics_rows.append(metrics)

    param_metrics = pd.DataFrame(parameter_metrics_rows)
    atomic_write_csv(param_metrics, FULL_OUTPUT / "parameter_recovery_metrics.csv")

    # --- Programmatic decisions ---
    decisions_rows = []
    for _, row in param_metrics.iterrows():
        decisions_rows.append({
            "check_id": f"recovery_{row['parameter']}",
            "parameter": row["parameter"],
            "decision": "PASS" if row["support"] in ("strong", "cautious") else "FAIL",
            "support_level": row["support"],
            "rule": f"rho>={RECOVERY_THRESHOLDS['strong']['rho']};bias_pct<={RECOVERY_THRESHOLDS['strong']['bias_pct']};cov50>={RECOVERY_THRESHOLDS['strong']['coverage']}",
            "observed_value": f"rho={row.get('rho', np.nan):.4g};bias_pct={row.get('bias_pct', np.nan):.4g};cov50={row.get('coverage_50', np.nan):.4g}",
            "programmatic": True,
            "contract_hash": contract["contract_hash"],
        })
    decisions = pd.DataFrame(decisions_rows)
    atomic_write_csv(decisions, FULL_OUTPUT / "recovery_decisions.csv")
    atomic_write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True),
        FULL_OUTPUT / "run_contract.json")

    n_complete = int(manifest_final["status"].eq("complete").sum())
    n_gen_fail = int(manifest_final["status"].eq("generation_failed").sum())
    n_finite_fail = int(manifest_final["status"].eq("finite_check_failed").sum())
    strong_params = int((param_metrics["support"] == "strong").sum())
    cautious_params = int((param_metrics["support"] == "cautious").sum())
    not_supported = int((param_metrics["support"] == "not_supported").sum())
    insufficient = int((param_metrics["support"] == "insufficient").sum())

    report_lines = [
        "# T60c-b T5 Full Parameter Recovery",
        "",
        f"- Truth bank: {len(truth_bank)} rows.",
        f"- Conditions: {len(SENSITIVITY_CONDITIONS)} ({', '.join(SENSITIVITY_CONDITIONS)}).",
        f"- Completed datasets: {n_complete}.",
        f"- Generation failures: {n_gen_fail}.",
        f"- Finite-check failures: {n_finite_fail}.",
        f"- Parameter support: strong={strong_params}, cautious={cautious_params}, "
        f"not_supported={not_supported}, insufficient={insufficient}.",
        "- Thresholds: see `recovery_config.json` and contract.",
        "- No clinical or diagnostic conclusions.",
    ]
    atomic_write_text("\n".join(report_lines) + "\n", FULL_OUTPUT / "analysis-report.md")

    return {
        "truth_count": len(truth_bank),
        "conditions": list(SENSITIVITY_CONDITIONS.keys()),
        "completed_datasets": n_complete,
        "generation_failures": n_gen_fail,
        "finite_failures": n_finite_fail,
        "strong": strong_params,
        "cautious": cautious_params,
        "not_supported": not_supported,
        "insufficient": insufficient,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "full":
        contract = full_contract(config, args.config)
        summary = run_full(config, contract, args.resume)
    else:
        contract = smoke_contract(config, args.config)
        summary = run_smoke(config, contract, args.resume)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
