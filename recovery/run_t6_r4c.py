#!/usr/bin/env python3
"""Run a separate T6 v2 synthetic-recovery pilot on observable-window truths.

The script samples gaze points from the unchanged v2 density, refits them
from fixed truth-blind starts, and writes pilot evidence only. It does not
change production code, configuration, the formal truth bank, or admission.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY = SCRIPT_DIR.parent
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from recovery.parameter_recovery_common import atomic_write_csv, atomic_write_text, stable_seed
from recovery.t6_r4c_model import (
    PARAMETER_NAMES,
    REFERENCE_TIME_MS,
    compute_component_weights_v2,
    compute_log_likelihood_v2,
    derive_pmax_preserving_pm,
    optimize_log1p,
    visual_angle_deg_to_px,
)

TASK_ID = "T60b-v2b-R4c"
HISTORICAL_MODEL_SHA256 = "026c91a5c49f1f491706e815937f0b6599994b9a7aa39a8c21629d16fa1dcae1"
DEFAULT_TRUTH_BANK = SCRIPT_DIR / "t6_r4c_truths.csv"
DEFAULT_CONFIG = SCRIPT_DIR / "t6_r4c_config.json"
DEFAULT_TRUTH_IDS = ("T6OW-01", "T6OW-03", "T6OW-05", "T6OW-08", "T6OW-09", "T6OW-12", "T6OW-14", "T6OW-16")
EMOTIONS = ("positive", "negative", "neutral")
HIT_COLUMNS = {"positive": "positive", "negative": "nagetive", "neutral": "neutral"}
AREA_COLUMNS = {
    "positive": "positive_area_px2",
    "negative": "negative_area_px2",
    "neutral": "neutral_area_px2",
}
RECTANGLES = (
    (100.0, 600.0, 340.0, 740.0),
    (710.0, 1210.0, 340.0, 740.0),
    (1320.0, 1820.0, 340.0, 740.0),
)
SCREEN_WIDTH = 1920.0
SCREEN_HEIGHT = 1080.0
HALF_CHI2_95 = 1.920729410347062
DISTINCT_DISTANCE = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-bank", type=Path, default=DEFAULT_TRUTH_BANK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-ids", nargs="+", default=list(DEFAULT_TRUTH_IDS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--samples-per-trial", type=int, default=61)
    parser.add_argument("--maxiter", type=int, default=250)
    return parser.parse_args()


def audit_bounds() -> dict[str, tuple[float, float]]:
    dc_lower = visual_angle_deg_to_px(0.5)
    dc_upper = visual_angle_deg_to_px(10.0)
    return {
        "Tmax1": (200.0, 2550.0), "Tmax2": (200.0, 2550.0), "Tmax3": (200.0, 2550.0),
        "Pmax1": (0.001, 10.0), "Pmax2": (0.001, 10.0), "Pmax3": (0.001, 10.0),
        "dc": (dc_lower, dc_upper),
    }


def truth_params(row: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in PARAMETER_NAMES}


def load_truths(path: Path, truth_ids: Sequence[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = ["truth_id", "design_hash", "input_contract_hash", *PARAMETER_NAMES]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"truth bank missing columns: {missing}")
    if frame["truth_id"].duplicated().any():
        raise ValueError("truth IDs must be unique")
    requested = list(dict.fromkeys(str(value) for value in truth_ids))
    unknown = sorted(set(requested) - set(frame["truth_id"].astype(str)))
    if unknown:
        raise ValueError(f"unknown truth IDs: {unknown}")
    selected = frame.set_index("truth_id").loc[requested].reset_index()
    if selected["design_hash"].nunique() != 1 or selected["input_contract_hash"].nunique() != 1:
        raise ValueError("selected truths do not share one design/input contract")
    bounds = audit_bounds()
    for name in PARAMETER_NAMES:
        values = pd.to_numeric(selected[name], errors="coerce")
        lower, upper = bounds[name]
        if not np.isfinite(values).all() or not values.between(lower, upper).all():
            raise ValueError(f"truth {name} is non-finite or outside pilot bounds")
        selected[name] = values
    return selected


def sample_truncated_central(rng: np.random.Generator, dc: float) -> tuple[float, float]:
    while True:
        x = float(rng.normal(SCREEN_WIDTH / 2.0, dc))
        y = float(rng.normal(SCREEN_HEIGHT / 2.0, dc))
        if 0.0 <= x <= SCREEN_WIDTH and 0.0 <= y <= SCREEN_HEIGHT:
            return x, y


def contains(rectangle: tuple[float, float, float, float], x: float, y: float) -> bool:
    left, right, top, bottom = rectangle
    return left <= x <= right and top <= y <= bottom


def simulate_dataset(
    params: Mapping[str, float], seed: int, samples_per_trial: int = 61, n_trials: int = 40
) -> dict[str, np.ndarray]:
    if samples_per_trial < 4 or n_trials < 1:
        raise ValueError("samples_per_trial must be >=4 and n_trials must be positive")
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 3000.0, samples_per_trial)
    rows: dict[str, list[float]] = {
        "gaze_x": [], "gaze_y": [], "trial_number": [], "picture": [], "outside_roi": [],
        "positive": [], "nagetive": [], "neutral": [],
        "positive_area_px2": [], "negative_area_px2": [], "neutral_area_px2": [],
        "t_from_picture_start": [],
    }
    rectangle_area = (RECTANGLES[0][1] - RECTANGLES[0][0]) * (RECTANGLES[0][3] - RECTANGLES[0][2])
    weights = compute_component_weights_v2(times, params)

    for trial in range(1, n_trials + 1):
        shift = (trial - 1) % 3
        emotion_rectangles = {
            emotion: RECTANGLES[(index + shift) % 3] for index, emotion in enumerate(EMOTIONS)
        }
        for time_index, time_ms in enumerate(times):
            probabilities = np.asarray(
                [weights["central"][time_index], *(weights[emotion][time_index] for emotion in EMOTIONS)]
            )
            component = int(rng.choice(4, p=probabilities))
            if component == 0:
                x, y = sample_truncated_central(rng, float(params["dc"]))
            else:
                rectangle = emotion_rectangles[EMOTIONS[component - 1]]
                x = float(rng.uniform(rectangle[0], rectangle[1]))
                y = float(rng.uniform(rectangle[2], rectangle[3]))
            hits = {emotion: float(contains(rectangle, x, y)) for emotion, rectangle in emotion_rectangles.items()}
            rows["gaze_x"].append(x)
            rows["gaze_y"].append(y)
            rows["trial_number"].append(float(trial))
            rows["picture"].append(1.0)
            rows["outside_roi"].append(float(sum(hits.values()) == 0.0))
            for emotion in EMOTIONS:
                rows[HIT_COLUMNS[emotion]].append(hits[emotion])
                rows[AREA_COLUMNS[emotion]].append(rectangle_area)
            rows["t_from_picture_start"].append(float(time_ms))
    return {
        name: np.asarray(values, dtype=np.int64 if name == "trial_number" else np.float64)
        for name, values in rows.items()
    }


def build_blind_starts(n_starts: int = 8) -> list[tuple[str, dict[str, float]]]:
    specifications = (
        ("blind_early", (600, 600, 600), (1 / 3, 1 / 3, 1 / 3), 0.35, 2.0),
        ("blind_middle", (1500, 1500, 1500), (1 / 3, 1 / 3, 1 / 3), 0.60, 5.0),
        ("blind_late", (2400, 2400, 2400), (1 / 3, 1 / 3, 1 / 3), 0.80, 7.0),
        ("blind_forward", (700, 1500, 2400), (0.40, 0.30, 0.30), 0.60, 5.0),
        ("blind_reverse", (2400, 1500, 700), (0.30, 0.40, 0.30), 0.60, 2.0),
        ("blind_rotated", (1500, 2400, 700), (0.30, 0.30, 0.40), 0.35, 7.0),
        ("blind_mixed_a", (900, 1800, 2300), (0.45, 0.35, 0.20), 0.75, 4.0),
        ("blind_mixed_b", (2100, 1000, 1700), (0.20, 0.45, 0.35), 0.45, 6.0),
    )
    if not 1 <= n_starts <= len(specifications):
        raise ValueError(f"n_starts must lie in [1, {len(specifications)}]")
    starts = []
    for name, tmax, pm, target_weight, dc_deg in specifications[:n_starts]:
        pmax = derive_pmax_preserving_pm(tmax, pm, target_weight, REFERENCE_TIME_MS)
        params = {
            "Tmax1": float(tmax[0]), "Tmax2": float(tmax[1]), "Tmax3": float(tmax[2]),
            "Pmax1": pmax[0], "Pmax2": pmax[1], "Pmax3": pmax[2],
            "dc": visual_angle_deg_to_px(dc_deg),
        }
        starts.append((name, params))
    vectors = {tuple(round(params[name], 12) for name in PARAMETER_NAMES) for _, params in starts}
    if len(vectors) != len(starts):
        raise ValueError("blind starts must be unique")
    return starts


def fit_one_dataset(
    data: Mapping[str, np.ndarray], starts: Sequence[tuple[str, dict[str, float]]], maxiter: int
) -> list[dict[str, Any]]:
    bounds = audit_bounds()
    log_likelihood = lambda params: compute_log_likelihood_v2(data, params)
    results = []
    for start_id, start_params in starts:
        fit = optimize_log1p(log_likelihood, start_params, bounds, maxiter=maxiter)
        row = {
            "start_id": start_id,
            "optimizer_success": bool(fit["optimizer_success"]),
            "valid_fit": bool(math.isfinite(float(fit["log_likelihood"]))),
            "log_likelihood": float(fit["log_likelihood"]),
            "message": fit["message"],
            "nit": fit["nit"],
            "nfev": fit["nfev"],
            "runtime_seconds": fit["runtime_seconds"],
        }
        row.update({f"fitted_{name}": float(fit["params"][name]) for name in PARAMETER_NAMES})
        results.append(row)
    return results


def normalized_vector(row: Mapping[str, Any], prefix: str = "fitted_") -> np.ndarray:
    bounds = audit_bounds()
    values = np.asarray([math.log1p(float(row[f"{prefix}{name}"])) for name in PARAMETER_NAMES])
    lower = np.asarray([math.log1p(bounds[name][0]) for name in PARAMETER_NAMES])
    upper = np.asarray([math.log1p(bounds[name][1]) for name in PARAMETER_NAMES])
    return (values - lower) / (upper - lower)


def summarize_datasets(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    best_rows = []
    for dataset_id, group in results.groupby("dataset_id", sort=False):
        finite = group[group["valid_fit"].astype(bool) & np.isfinite(group["log_likelihood"])].copy()
        if finite.empty:
            summaries.append({"dataset_id": dataset_id, "finite_starts": 0, "successful_starts": 0})
            continue
        best = finite.loc[finite["log_likelihood"].idxmax()].copy()
        best_ll = float(best["log_likelihood"])
        finite["delta_nll"] = best_ll - finite["log_likelihood"]
        near = finite[finite["delta_nll"] <= HALF_CHI2_95]
        distinct = False
        near_rows = list(near.to_dict("records"))
        for left_index in range(len(near_rows)):
            for right_index in range(left_index + 1, len(near_rows)):
                distance = float(np.max(np.abs(normalized_vector(near_rows[left_index]) - normalized_vector(near_rows[right_index]))))
                distinct |= distance >= DISTINCT_DISTANCE
        successful = finite[finite["optimizer_success"].astype(bool)]
        basin_fraction = float((successful["delta_nll"] <= 0.5).mean()) if len(successful) else 0.0
        fitted_pmax = np.asarray([best[f"fitted_Pmax{i}"] for i in range(1, 4)], dtype=float)
        fitted_pm = fitted_pmax / fitted_pmax.sum()
        truth_pmax = np.asarray([best[f"truth_Pmax{i}"] for i in range(1, 4)], dtype=float)
        truth_pm = truth_pmax / truth_pmax.sum()
        for index, emotion in enumerate(EMOTIONS):
            best[f"truth_pm_{emotion}"] = truth_pm[index]
            best[f"fitted_pm_{emotion}"] = fitted_pm[index]
            best[f"abs_error_pm_{emotion}"] = abs(fitted_pm[index] - truth_pm[index])
        for name in PARAMETER_NAMES:
            truth_value = float(best[f"truth_{name}"])
            fitted_value = float(best[f"fitted_{name}"])
            best[f"relative_error_{name}"] = abs(fitted_value - truth_value) / truth_value
        best["max_tmax_relative_error"] = max(best[f"relative_error_Tmax{i}"] for i in range(1, 4))
        best["max_pmax_relative_error"] = max(best[f"relative_error_Pmax{i}"] for i in range(1, 4))
        best["max_pm_absolute_error"] = max(best[f"abs_error_pm_{emotion}"] for emotion in EMOTIONS)
        best["dc_relative_error"] = best["relative_error_dc"]
        best["scenario_pass"] = bool(
            best["max_tmax_relative_error"] <= 0.25
            and best["max_pmax_relative_error"] <= 0.30
            and best["max_pm_absolute_error"] <= 0.10
            and best["dc_relative_error"] <= 0.25
        )
        best_rows.append(best.to_dict())
        summaries.append(
            {
                "dataset_id": dataset_id,
                "truth_id": best["truth_id"],
                "finite_starts": len(finite),
                "successful_starts": len(successful),
                "best_basin_fraction": basin_fraction,
                "near_support_starts": len(near),
                "distinct_near_equivalent_solution": distinct,
                "best_log_likelihood": best_ll,
            }
        )
    return pd.DataFrame(best_rows), pd.DataFrame(summaries)


def safe_spearman(truth: pd.Series, fitted: pd.Series) -> float:
    if truth.nunique() < 2 or fitted.nunique() < 2:
        return math.nan
    result = spearmanr(truth, fitted)
    value = getattr(result, "statistic", None)
    if value is None:
        value = result.correlation
    return float(value)


def recovery_metrics(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    definitions = [(name, name, "relative") for name in PARAMETER_NAMES]
    definitions.extend((f"pm_{emotion}", f"pm_{emotion}", "absolute") for emotion in EMOTIONS)
    for label, suffix, error_scale in definitions:
        truth = best[f"truth_{suffix}"].astype(float)
        fitted = best[f"fitted_{suffix}"].astype(float)
        error = fitted - truth
        absolute = error.abs()
        relative = absolute / truth.abs().clip(lower=1e-12)
        rows.append(
            {
                "parameter": label,
                "n": len(best),
                "n_unique_truth": truth.nunique(),
                "bias": float(error.mean()),
                "mae": float(absolute.mean()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "median_relative_error": float(relative.median()),
                "spearman_rho": safe_spearman(truth, fitted),
                "primary_error_scale": error_scale,
            }
        )
    return pd.DataFrame(rows)


def make_decision(best: pd.DataFrame, summaries: pd.DataFrame, metrics: pd.DataFrame, planned: int) -> pd.DataFrame:
    integrity = bool(
        len(best) == planned
        and len(summaries) == planned
        and (summaries["finite_starts"] >= 4).all()
    )
    metric = metrics.set_index("parameter")
    tmax_pass = all(
        metric.loc[f"Tmax{i}", "median_relative_error"] <= 0.25
        and metric.loc[f"Tmax{i}", "spearman_rho"] >= 0.70
        for i in range(1, 4)
    )
    pm_pass = all(
        metric.loc[f"pm_{emotion}", "mae"] <= 0.10
        and metric.loc[f"pm_{emotion}", "spearman_rho"] >= 0.70
        for emotion in EMOTIONS
    )
    dc_pass = bool(
        metric.loc["dc", "median_relative_error"] <= 0.25
        and metric.loc["dc", "spearman_rho"] >= 0.70
    )
    pmax_pass = all(metric.loc[f"Pmax{i}", "median_relative_error"] <= 0.30 for i in range(1, 4))
    distinct_rate = float(summaries["distinct_near_equivalent_solution"].mean()) if len(summaries) else 1.0
    if not integrity:
        overall = "SYNTHETIC_RECOVERY_PILOT_INCONCLUSIVE"
    elif tmax_pass and pm_pass and dc_pass and pmax_pass and distinct_rate <= 0.25:
        overall = "SYNTHETIC_RECOVERY_PILOT_SUPPORTS_PARAMETERS"
    elif tmax_pass or pm_pass or dc_pass or pmax_pass:
        overall = "SYNTHETIC_RECOVERY_PILOT_MIXED"
    else:
        overall = "SYNTHETIC_RECOVERY_PILOT_DOES_NOT_SUPPORT_PARAMETERS"
    return pd.DataFrame(
        [
            {
                "task_id": TASK_ID,
                "decision": overall,
                "execution_integrity": integrity,
                "tmax_pass": tmax_pass,
                "pm_pass": pm_pass,
                "pmax_pass": pmax_pass,
                "dc_pass": dc_pass,
                "scenario_pass_count": int(best["scenario_pass"].sum()) if len(best) else 0,
                "scenario_count": len(best),
                "distinct_solution_rate": distinct_rate,
                "parameter_recovery_admission": "UNCHANGED_BLOCKED",
            }
        ]
    )


def render_report(best: pd.DataFrame, summaries: pd.DataFrame, metrics: pd.DataFrame, decision: pd.DataFrame) -> str:
    row = decision.iloc[0]
    key = metrics[metrics["parameter"].isin(["Tmax1", "Tmax2", "Tmax3", "pm_positive", "pm_negative", "pm_neutral", "dc"])]
    table = ["| Parameter | MAE | Median relative error | Spearman rho |", "|---|---:|---:|---:|"]
    for metric in key.itertuples():
        table.append(f"| {metric.parameter} | {metric.mae:.6g} | {metric.median_relative_error:.3f} | {metric.spearman_rho:.3f} |")
    return f"""# T6 可观测窗口合成参数恢复试运行

## 1. 模型答题方式

使用未修改的T6 v2概率密度，从预先知道的参数生成40个trial、每trial 61个时间点的合成gaze。随后从8个固定且不读取真值的起点重新拟合，并保留似然最高的答案。拟合仅使用审计边界：Tmax 200–2550 ms、Pmax 0.001–10、dc 0.5°–10°。

## 2. 完成情况

- 合成并拟合的数据集：{len(best)}。
- 每个数据集计划起点：{int(summaries['finite_starts'].max()) if len(summaries) else 0}。
- 完整通过全部单场景误差标准：{int(best['scenario_pass'].sum())}/{len(best)}。
- 出现明显不同近似等价解的数据集：{int(summaries['distinct_near_equivalent_solution'].sum())}/{len(summaries)}。

## 3. 主要恢复指标

{chr(10).join(table)}

预声明标准：Tmax中位相对误差≤25%且rho≥0.70；pm MAE≤0.10且rho≥0.70；Pmax中位相对误差≤30%；dc中位相对误差≤25%且rho≥0.70。

## 4. 判断

- 总体结论：`{row['decision']}`。
- Tmax：{'通过' if row['tmax_pass'] else '未通过'}。
- pm：{'通过' if row['pm_pass'] else '未通过'}。
- Pmax：{'通过' if row['pmax_pass'] else '未通过'}。
- dc：{'通过' if row['dc_pass'] else '未通过'}。
- Parameter Recovery正式准入：`UNCHANGED_BLOCKED`。

## 5. 限制

这是理想条件下的模型内恢复：生成模型和拟合模型完全相同，且使用规则矩形刺激和独立抽样。若理想条件下仍恢复不佳，支持结构性耦合或有限窗口信息不足；若通过，也仍需使用真实任务时间采样、噪声和缺失机制扩大验证。
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 1 or args.samples_per_trial < 4 or args.maxiter < 1:
        raise ValueError("repeats, samples_per_trial, and maxiter must be positive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    truths = load_truths(args.truth_bank, args.truth_ids)
    starts = build_blind_starts(args.n_starts)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "recovery_start_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be empty")
    rows = []
    manifest_rows = []

    for truth in truths.to_dict("records"):
        params = truth_params(truth)
        for repeat in range(1, args.repeats + 1):
            dataset_id = f"{truth['truth_id']}-R{repeat:02d}"
            seed = stable_seed(int(config["random_seed"]), str(truth["design_hash"]), dataset_id, "synthetic")
            data = simulate_dataset(params, seed, args.samples_per_trial)
            manifest_rows.append(
                {
                    "dataset_id": dataset_id, "truth_id": truth["truth_id"], "repeat": repeat,
                    "seed": seed, "n_trials": 40, "samples_per_trial": args.samples_per_trial,
                    "n_observations": len(data["gaze_x"]), "design_hash": truth["design_hash"],
                    "input_contract_hash": truth["input_contract_hash"], "model_code_sha256": HISTORICAL_MODEL_SHA256,
                }
            )
            print(f"[{TASK_ID}] {dataset_id}: fitting {len(starts)} starts", flush=True)
            for fit in fit_one_dataset(data, starts, args.maxiter):
                fit["run_id"] = f"{dataset_id}:{fit['start_id']}"
                fit["dataset_id"] = dataset_id
                fit["truth_id"] = truth["truth_id"]
                fit["repeat"] = repeat
                fit["seed"] = seed
                fit["design_hash"] = truth["design_hash"]
                fit["input_contract_hash"] = truth["input_contract_hash"]
                fit.update({f"truth_{name}": params[name] for name in PARAMETER_NAMES})
                rows.append(fit)
                atomic_write_csv(pd.DataFrame(rows).sort_values(["dataset_id", "start_id"]), result_path)

    manifest = pd.DataFrame(manifest_rows).sort_values("dataset_id")
    results = pd.DataFrame(rows).sort_values(["dataset_id", "start_id"])
    expected_runs = len(manifest) * len(starts)
    if len(results) != expected_runs or results["run_id"].nunique() != expected_runs:
        raise ValueError(f"incomplete pilot results: {len(results)}/{expected_runs}")
    best, summaries = summarize_datasets(results)
    metrics = recovery_metrics(best)
    decision = make_decision(best, summaries, metrics, len(manifest))
    atomic_write_csv(manifest, output_dir / "simulation_manifest.csv")
    atomic_write_csv(best, output_dir / "best_recovery_results.csv")
    atomic_write_csv(summaries, output_dir / "dataset_recovery_summary.csv")
    atomic_write_csv(metrics, output_dir / "parameter_recovery_metrics.csv")
    atomic_write_csv(decision, output_dir / "audit_decisions.csv")
    atomic_write_text(render_report(best, summaries, metrics, decision), output_dir / "analysis-report.md")
    print(decision.to_string(index=False), flush=True)
    return {"datasets": len(best), "runs": len(results), "decision": decision.iloc[0]["decision"]}


def main() -> None:
    result = run(parse_args())
    print(f"{TASK_ID}: {result}", flush=True)


if __name__ == "__main__":
    main()
