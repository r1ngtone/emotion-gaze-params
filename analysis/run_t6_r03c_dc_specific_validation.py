#!/usr/bin/env python3
"""Run statistics, fit diagnostics, figures, and report for T6 R03c."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-r03c")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
T6_ROOT = REPO_ROOT / "multipic_eye" / "多图" / "t6_devide"
if str(T6_ROOT) not in sys.path:
    sys.path.insert(0, str(T6_ROOT))

from corecode.pipeline import (  # noqa: E402
    DEFAULT_BOUNDS,
    annotate_spatial_and_emotion,
    build_base_frame,
    compute_log_likelihood,
    prepare_model_data,
)


PARAM_CSV = SCRIPT_DIR.parents[2] / "data" / "单图多图参数_量表匹配总表_合并_N80.csv"
R03B_DIR = SCRIPT_DIR / "core_model_free_tmax_stability_R03b_N80"
REPRO_CSV = R03B_DIR / "T6_Tmax_production_reproduction_subject_N80.csv"
BEST_CSV = R03B_DIR / "T6_Tmax_best_fits_N80.csv"
RUNS_CSV = R03B_DIR / "T6_Tmax_multistart_runs_N80.csv"
SPLIT_CSV = R03B_DIR / "T6_Tmax_split_half_subject_N80.csv"
OUTPUT_DIR = SCRIPT_DIR / "core_model_free_dc_specific_validation_R03c_N80"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_CSV = OUTPUT_DIR / "dc_specific_direct_metrics_N80.csv"
REPORT_PATH = OUTPUT_DIR / "2026-06-22--t6-model-free--r03c--dc-specific-validation-report.md"

RANDOM_SEED = 20260622
BOOTSTRAP_REPS = 2000
NEAR_OPTIMAL_LL_DELTA = 1.0
DC_BOUNDS = (1.0, 2000.0)
METRICS = [
    "center_gaze_dispersion_px",
    "mean_distance_to_center_px",
    "early_phase_center_escape_radius_px",
    "transition_endpoint_dispersion_px",
    "within_AOI_gaze_dispersion_px",
    "residual_spatial_error_px",
]
METRIC_LABELS = {
    "center_gaze_dispersion_px": "Center gaze dispersion",
    "mean_distance_to_center_px": "Mean distance to center",
    "early_phase_center_escape_radius_px": "Early center escape radius",
    "transition_endpoint_dispersion_px": "Transition endpoint dispersion",
    "within_AOI_gaze_dispersion_px": "Within-AOI gaze dispersion",
    "residual_spatial_error_px": "Residual spatial error",
}
QUALITY_METRICS = [
    "screen_valid_points",
    "valid_trial_count",
    "missing_rate",
    "out_of_screen_rate",
]


def bh_fdr(values: Iterable[float]) -> list[float]:
    values = list(values)
    adjusted = [math.nan] * len(values)
    valid = sorted(((index, value) for index, value in enumerate(values) if pd.notna(value)), key=lambda x: x[1])
    running = 1.0
    for rank, (index, value) in reversed(list(enumerate(valid, start=1))):
        running = min(running, value * len(valid) / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return math.nan, math.nan, len(frame)
    result = stats.spearmanr(frame["x"], frame["y"])
    return float(result[0]), float(result[1]), len(frame)


def bootstrap_ci(x: pd.Series, y: pd.Series, seed: int) -> tuple[float, float]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 4:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    xa = frame["x"].to_numpy()
    ya = frame["y"].to_numpy()
    samples = rng.integers(0, len(frame), size=(BOOTSTRAP_REPS, len(frame)))
    ranked_x = stats.rankdata(xa[samples], axis=1)
    ranked_y = stats.rankdata(ya[samples], axis=1)
    ranked_x = ranked_x - ranked_x.mean(axis=1, keepdims=True)
    ranked_y = ranked_y - ranked_y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(ranked_x ** 2, axis=1) * np.sum(ranked_y ** 2, axis=1))
    values = np.divide(
        np.sum(ranked_x * ranked_y, axis=1), denominator,
        out=np.full(BOOTSTRAP_REPS, np.nan), where=denominator > 0,
    )
    values = values[np.isfinite(values)]
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if len(values) else (math.nan, math.nan)


def visual_angle_to_px(degrees: float) -> float:
    cm = 2.0 * 57.0 * math.tan(math.radians(degrees) / 2.0)
    return cm * 1920.0 / 34.5


def audit_dc_scale(params: pd.DataFrame, repro: pd.DataFrame) -> pd.DataFrame:
    merged = params[["subject_id", "dc_多图"]].merge(
        repro[["subject_id", "dc"]].rename(columns={"dc": "dc_default_refit_px"}), on="subject_id", how="inner"
    )
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        original = float(row.dc_多图)
        refit = float(row.dc_default_refit_px)
        converted = visual_angle_to_px(original)
        keep_error = abs(math.log(max(original, 1e-12) / refit))
        convert_error = abs(math.log(max(converted, 1e-12) / refit))
        converted_wins = convert_error < keep_error
        harmonized = converted if converted_wins else original
        margin = abs(keep_error - convert_error)
        rows.append({
            "subject_id": str(row.subject_id),
            "dc_original": original,
            "dc_default_refit_px": refit,
            "dc_if_angle_converted_px": converted,
            "keep_log_error": keep_error,
            "convert_log_error": convert_error,
            "dc_source_scale": "visual_angle_deg" if converted_wins else "pixel",
            "dc_scale_ambiguous": margin < math.log(2.0),
            "dc_harmonized_px": harmonized,
            "dc_harmonized_to_refit_ratio": harmonized / refit,
        })
    return pd.DataFrame(rows)


def prepare_model_data_from_paths(eye_path: str, record_path: str) -> dict[str, np.ndarray]:
    eye = pd.read_csv(eye_path, encoding="utf-8-sig")
    records = pd.read_csv(record_path, encoding="utf-8-sig")
    base, _ = build_base_frame(eye)
    annotated = annotate_spatial_and_emotion(base, records)
    _, data = prepare_model_data(annotated)
    return data


def fixed_dc_fit(data: dict[str, np.ndarray], start: pd.Series, fixed_dc: float) -> tuple[float, bool, str]:
    names = ["Tmax1", "Tmax2", "Tmax3", "Pmax1", "Pmax2", "Pmax3"]
    initial = np.array([float(start[f"{name}_best"] if f"{name}_best" in start.index else start[name]) for name in names])
    bounds = [DEFAULT_BOUNDS[name] for name in names]

    def objective(theta: np.ndarray) -> float:
        values = {name: float(value) for name, value in zip(names, theta)}
        values["dc"] = fixed_dc
        ll = compute_log_likelihood(data, values)
        return -ll if np.isfinite(ll) else 1e100

    result = minimize(objective, x0=initial, bounds=bounds, method="L-BFGS-B")
    if not result.success:
        retry = minimize(objective, x0=result.x, bounds=bounds, method="Powell")
        if retry.success or retry.fun < result.fun:
            result = retry
    return float(-result.fun), bool(result.success), str(result.message)


def run_fit_diagnostics(frame: pd.DataFrame, best: pd.DataFrame, fixed_dc: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    best_full = best[best["split"] == "full"].copy() if "split" in best.columns else best.copy()
    best_full = best_full.sort_values("best_log_likelihood", ascending=False).drop_duplicates("subject_id")
    merged = frame.merge(best_full, on="subject_id", how="inner", suffixes=("", "_best"))
    for index, row in merged.iterrows():
        data = prepare_model_data_from_paths(str(row["raw_eyetrack_path"]), str(row["raw_records_path"]))
        production_params = {
            "Tmax1": float(row["Tmax1"]),
            "Tmax2": float(row["Tmax2"]),
            "Tmax3": float(row["Tmax3"]),
            "Pmax1": float(row["Pmax_positive_多图"]),
            "Pmax2": float(row["Pmax_negative_多图"]),
            "Pmax3": float(row["Pmax_neutral_多图"]),
            "dc": float(row["dc_harmonized_px"]),
        }
        production_ll = compute_log_likelihood(data, production_params)
        fixed_ll, success, message = fixed_dc_fit(data, row, fixed_dc)
        free_ll = float(row["best_log_likelihood"])
        n_obs = int(len(data["timestamp"]))
        rows.append({
            "subject_id": str(row["subject_id"]),
            "production_nll": -production_ll,
            "production_mean_nll": -production_ll / n_obs,
            "free_multistart_nll": -free_ll,
            "free_multistart_mean_nll": -free_ll / n_obs,
            "fixed_dc_px": fixed_dc,
            "fixed_dc_success": success,
            "fixed_dc_message": message,
            "fixed_dc_nll": -fixed_ll,
            "fixed_dc_mean_nll": -fixed_ll / n_obs,
            "delta_nll_fixed_minus_free": free_ll - fixed_ll,
            "aic_free_dc": 2 * 7 - 2 * free_ll,
            "aic_fixed_dc": 2 * 6 - 2 * fixed_ll,
            "delta_aic_fixed_minus_free": (2 * 6 - 2 * fixed_ll) - (2 * 7 - 2 * free_ll),
            "model_n_observations": n_obs,
        })
        print(f"[fit {index + 1:02d}/{len(merged)}] subject_id={row['subject_id']} fixed_success={success}")
    return pd.DataFrame(rows)


def metric_reliability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, metric in enumerate(METRICS):
        rho, p_value, n_valid = spearman(frame[f"{metric}_odd"], frame[f"{metric}_even"])
        ci_low, ci_high = bootstrap_ci(frame[f"{metric}_odd"], frame[f"{metric}_even"], RANDOM_SEED + index)
        if rho >= 0.50 and ci_low > 0:
            label = "stable"
        elif rho >= 0.30 or ci_low > 0:
            label = "cautious"
        else:
            label = "unstable"
        rows.append({"metric": metric, "rho": rho, "p_value": p_value, "ci95_low": ci_low, "ci95_high": ci_high, "n_valid": n_valid, "stability_level": label})
    output = pd.DataFrame(rows)
    output["p_bh_fdr"] = bh_fdr(output["p_value"])
    return output


def direct_correlations(frame: pd.DataFrame, reliability: pd.DataFrame) -> pd.DataFrame:
    stability = reliability.set_index("metric")["stability_level"].to_dict()
    rows = []
    for index, metric in enumerate(METRICS):
        rho, p_value, n_valid = spearman(frame["dc_harmonized_px"], frame[metric])
        ci_low, ci_high = bootstrap_ci(frame["dc_harmonized_px"], frame[metric], RANDOM_SEED + 100 + index)
        rho_default, p_default, n_default = spearman(frame["dc_default_refit_px"], frame[metric])
        rho_best, p_best, n_best = spearman(frame["dc_multistart_best_px"], frame[metric])
        non_extreme = frame[~frame["dc_extreme_mad"]]
        rho_non_extreme, p_non_extreme, n_non_extreme = spearman(non_extreme["dc_harmonized_px"], non_extreme[metric])
        partial_rho, partial_p, partial_n = partial_rank_correlation(
            frame, "dc_harmonized_px", metric, ["screen_valid_points", "missing_rate", "out_of_screen_rate"]
        )
        rows.append({
            "direct_metric": metric,
            "rho": rho,
            "p_value": p_value,
            "rho_ci95_low": ci_low,
            "rho_ci95_high": ci_high,
            "n_valid": n_valid,
            "effect_direction": "positive" if rho > 0 else "negative" if rho < 0 else "zero",
            "expected_direction": "positive",
            "metric_stability_level": stability.get(metric, "unstable"),
            "rho_default_refit_sensitivity": rho_default,
            "p_default_refit_sensitivity": p_default,
            "n_default_refit_sensitivity": n_default,
            "rho_multistart_best_sensitivity": rho_best,
            "p_multistart_best_sensitivity": p_best,
            "n_multistart_best_sensitivity": n_best,
            "rho_excluding_mad_extremes": rho_non_extreme,
            "p_excluding_mad_extremes": p_non_extreme,
            "n_excluding_mad_extremes": n_non_extreme,
            "rho_quality_adjusted_rank_partial": partial_rho,
            "p_quality_adjusted_rank_partial": partial_p,
            "n_quality_adjusted_rank_partial": partial_n,
        })
    output = pd.DataFrame(rows)
    output["p_bh_fdr"] = bh_fdr(output["p_value"])
    levels = []
    for row in output.itertuples(index=False):
        sensitivity_direction = row.rho_default_refit_sensitivity > 0 and row.rho_multistart_best_sensitivity > 0
        strong = row.rho >= 0.30 and row.p_bh_fdr < 0.05 and row.rho_ci95_low > 0 and row.metric_stability_level == "stable" and sensitivity_direction
        cautious = row.rho > 0 and (row.p_bh_fdr < 0.05 or row.rho_ci95_low > 0 or abs(row.rho) >= 0.30)
        levels.append("strong" if strong else "cautious" if cautious else "unsupported")
    output["interpretation_level"] = levels
    return output


def partial_rank_correlation(
    frame: pd.DataFrame,
    x_name: str,
    y_name: str,
    covariates: list[str],
) -> tuple[float, float, int]:
    columns = [x_name, y_name, *covariates]
    data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    varying_covariates = [name for name in covariates if data[name].nunique() > 1]
    if len(data) < 5 or data[x_name].nunique() < 2 or data[y_name].nunique() < 2:
        return math.nan, math.nan, len(data)
    ranked = data[[x_name, y_name, *varying_covariates]].rank(method="average")
    design = np.column_stack([np.ones(len(ranked)), ranked[varying_covariates].to_numpy()])
    x_residual = ranked[x_name].to_numpy() - design @ np.linalg.lstsq(design, ranked[x_name].to_numpy(), rcond=None)[0]
    y_residual = ranked[y_name].to_numpy() - design @ np.linalg.lstsq(design, ranked[y_name].to_numpy(), rcond=None)[0]
    result = stats.pearsonr(x_residual, y_residual)
    return float(result[0]), float(result[1]), len(data)


def correlation_rows(frame: pd.DataFrame, variables: list[str], scope: str) -> pd.DataFrame:
    rows = []
    for index, variable in enumerate(variables):
        rho, p_value, n_valid = spearman(frame["dc_harmonized_px"], frame[variable])
        ci_low, ci_high = bootstrap_ci(frame["dc_harmonized_px"], frame[variable], RANDOM_SEED + 300 + index)
        rows.append({
            "analysis_scope": scope,
            "test_type": "spearman",
            "x_variable": "dc_harmonized_px",
            "y_variable": variable,
            "rho": rho,
            "p_value": p_value,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "n_valid": n_valid,
        })
    output = pd.DataFrame(rows)
    output["p_bh_fdr"] = bh_fdr(output["p_value"])
    return output


def compensation_diagnostics(frame: pd.DataFrame, runs: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    variables = ["pm_positive", "pm_negative", "pm_neutral", "Tmax1", "Tmax2", "Tmax3", "production_nll", "production_mean_nll"]
    output = correlation_rows(frame, variables, "across_subject")
    successful = runs[(runs["split"] == "full") & runs["success"] & runs["best_log_likelihood"].notna()].copy()
    successful["best_ll"] = successful.groupby("subject_id")["best_log_likelihood"].transform("max")
    near = successful[successful["best_ll"] - successful["best_log_likelihood"] <= NEAR_OPTIMAL_LL_DELTA].copy()
    trade_rows = []
    for variable in ["pm_positive", "pm_negative", "pm_neutral", "Tmax1", "Tmax2", "Tmax3"]:
        subset = near[["subject_id", "dc", variable]].dropna().copy()
        subset["dc_centered"] = subset["dc"] - subset.groupby("subject_id")["dc"].transform("mean")
        subset["variable_centered"] = subset[variable] - subset.groupby("subject_id")[variable].transform("mean")
        informative = subset.groupby("subject_id").filter(lambda group: len(group) >= 2 and group["dc"].nunique() >= 2)
        rho, p_value, n_valid = spearman(informative["dc_centered"], informative["variable_centered"])
        trade_rows.append({
            "analysis_scope": "within_subject_near_optimal_centered",
            "test_type": "spearman",
            "x_variable": "dc_centered",
            "y_variable": f"{variable}_centered",
            "rho": rho,
            "p_value": p_value,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "n_valid": n_valid,
            "p_bh_fdr": math.nan,
        })
    trade = pd.DataFrame(trade_rows)
    trade["p_bh_fdr"] = bh_fdr(trade["p_value"])
    split_rho, split_p, split_n = spearman(split["dc_odd"], split["dc_even"])
    split_ci = bootstrap_ci(split["dc_odd"], split["dc_even"], RANDOM_SEED + 500)
    split_row = pd.DataFrame([{
        "analysis_scope": "dc_split_half",
        "test_type": "spearman",
        "x_variable": "dc_odd",
        "y_variable": "dc_even",
        "rho": split_rho,
        "p_value": split_p,
        "ci95_low": split_ci[0],
        "ci95_high": split_ci[1],
        "n_valid": split_n,
        "p_bh_fdr": split_p,
    }])
    return pd.concat([output, trade, split_row], ignore_index=True)


def mad_flags(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    mad = np.median(np.abs(numeric.dropna() - median))
    if mad == 0:
        return pd.Series(False, index=values.index)
    score = 0.6745 * (numeric - median) / mad
    return score.abs() > 3.5


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def make_figures(frame: pd.DataFrame, direct: pd.DataFrame, reliability: pd.DataFrame, quality: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(frame["dc_harmonized_px"], kde=True, ax=axes[0], color="#0072B2")
    axes[0].set(title="Harmonized dc distribution", xlabel="dc (px)")
    sns.boxplot(x=frame["dc_harmonized_px"], ax=axes[1], color="#56B4E9")
    axes[1].set(title="dc spread and extremes", xlabel="dc (px)")
    save_figure(fig, "R03c-1_dc_distribution")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    direct_map = direct.set_index("direct_metric")
    for ax, metric in zip(axes.flat, METRICS):
        sns.regplot(data=frame, x="dc_harmonized_px", y=metric, lowess=True, scatter_kws={"s": 25, "alpha": 0.7}, line_kws={"color": "#D55E00"}, ax=ax)
        row = direct_map.loc[metric]
        ax.set_title(f"{METRIC_LABELS[metric]}\nrho={row.rho:.3f}, q={row.p_bh_fdr:.3g}, {row.interpretation_level}")
        ax.set_xlabel("dc (px)")
    fig.tight_layout()
    save_figure(fig, "R03c-2_dc_direct_metric_scatter_grid")

    matrix_vars = ["dc_harmonized_px", "pm_positive", "pm_negative", "pm_neutral", "Tmax1", "Tmax2", "Tmax3", "production_mean_nll"]
    corr = frame[matrix_vars].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", ax=ax)
    ax.set_title("dc, pm, Tmax, and mean NLL Spearman matrix")
    save_figure(fig, "R03c-3_parameter_compensation_matrix")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.flat, QUALITY_METRICS):
        sns.regplot(data=frame, x="dc_harmonized_px", y=metric, lowess=True, scatter_kws={"s": 24, "alpha": 0.7}, line_kws={"color": "#009E73"}, ax=ax)
        qrow = quality[quality["y_variable"] == metric].iloc[0]
        ax.set_title(f"{metric}: rho={qrow.rho:.3f}, q={qrow.p_bh_fdr:.3g}")
    fig.tight_layout()
    save_figure(fig, "R03c-4_dc_quality_relationships")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = np.where(frame["dc_extreme_mad"], "#D55E00", "#0072B2")
    axes[0].scatter(np.arange(len(frame)), frame["dc_harmonized_px"], c=colors, s=28)
    axes[0].axhline(DC_BOUNDS[0], color="black", ls="--", lw=1)
    axes[0].axhline(DC_BOUNDS[1], color="black", ls="--", lw=1)
    axes[0].set(title="Boundary and MAD extreme audit", xlabel="Subject order", ylabel="dc (px)")
    axes[1].scatter(frame["dc_harmonized_px"], frame["dc_default_refit_px"], c=colors, s=28)
    limit = max(frame["dc_harmonized_px"].max(), frame["dc_default_refit_px"].max())
    axes[1].plot([0, limit], [0, limit], color="black", ls="--", lw=1)
    axes[1].set(title="Production harmonized vs default refit", xlabel="Harmonized production dc", ylabel="Default-refit dc")
    save_figure(fig, "R03c-5_boundary_extreme_scale_audit")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    rel_map = reliability.set_index("metric")
    for ax, metric in zip(axes.flat, METRICS):
        ax.scatter(frame[f"{metric}_odd"], frame[f"{metric}_even"], s=24, alpha=0.7)
        row = rel_map.loc[metric]
        ax.set_title(f"{METRIC_LABELS[metric]}\nrho={row.rho:.3f}, {row.stability_level}")
        ax.set_xlabel("Odd trials")
        ax.set_ylabel("Even trials")
    fig.tight_layout()
    save_figure(fig, "R03c-6_direct_metric_split_half")


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.4f}"


def write_report(
    frame: pd.DataFrame,
    scale: pd.DataFrame,
    direct: pd.DataFrame,
    reliability: pd.DataFrame,
    compensation: pd.DataFrame,
    quality: pd.DataFrame,
    fit: pd.DataFrame,
) -> None:
    strong = direct[direct["interpretation_level"] == "strong"]
    cautious = direct[direct["interpretation_level"] == "cautious"]
    supported = direct[direct["interpretation_level"] != "unsupported"].sort_values("rho", ascending=False)
    strongest = supported.iloc[0] if not supported.empty else direct.sort_values("rho", ascending=False).iloc[0]
    split_row = compensation[compensation["analysis_scope"] == "dc_split_half"].iloc[0]
    trade = compensation[compensation["analysis_scope"] == "within_subject_near_optimal_centered"]
    trade_signal = trade[(trade["p_bh_fdr"] < 0.05) & (trade["rho"].abs() >= 0.30)]
    quality_signal = quality[quality["p_bh_fdr"] < 0.05]
    quality_robust = direct[(direct["rho_quality_adjusted_rank_partial"] > 0) & (direct["p_quality_adjusted_rank_partial"] < 0.05)]
    extreme_robust = direct[(direct["rho_excluding_mad_extremes"] > 0) & (direct["p_excluding_mad_extremes"] < 0.05)]
    successful_fit = fit[fit["fixed_dc_success"]].copy()
    fixed_preferred = int((successful_fit["delta_aic_fixed_minus_free"] < 0).sum())
    free_preferred = int((successful_fit["delta_aic_fixed_minus_free"] >= 0).sum())
    if len(strong) and split_row.rho >= 0.50 and trade_signal.empty:
        recommendation = "保留 `dc`，但仍限定为范式内空间尺度参数。"
    elif len(strong) + len(cautious) and (split_row.rho < 0.50 or not trade_signal.empty):
        recommendation = "将 `dc` 降级保留为探索性空间参数，并优先重构中心宽度、转移不确定性和 AOI 内扩散为可分离参数。"
    elif fixed_preferred >= free_preferred:
        recommendation = "当前先固定 `dc` 为 cohort 常量；在重构空间观测层前不用于个体解释。"
    else:
        recommendation = "暂保留 `dc` 仅用于模型拟合，不赋予直接行为解释；下一步重构空间参数后再决定删除。"
    stable_metrics = int((reliability["stability_level"] == "stable").sum())
    converted = int((scale["dc_source_scale"] == "visual_angle_deg").sum())
    ambiguous = int(scale["dc_scale_ambiguous"].sum())
    lines = [
        "# T6 R03c — `dc` 专项 direct behavioral correspondence 验证",
        "",
        "## 执行摘要",
        "",
        f"本分析纳入 {frame['subject_id'].nunique()} 名正常样本参与者。原参数表中 {converted}/80 个 `dc` 被识别为视觉角度尺度并逆变换为像素，{ambiguous}/80 个尺度分类存在歧义。主分析统一使用像素尺度，并用默认重拟合与 multi-start 最佳解复核。",
        "",
        f"六项专项指标中 strong={len(strong)}、cautious={len(cautious)}、unsupported={len(direct)-len(strong)-len(cautious)}；split-half 稳定指标为 {stable_metrics}/6。最强关系为 `{strongest.direct_metric}`（rho={strongest.rho:.3f}, BH-FDR={format_p(strongest.p_bh_fdr)}, 95% CI [{strongest.rho_ci95_low:.3f}, {strongest.rho_ci95_high:.3f}]）。",
        "",
        "## 方法边界",
        "",
        "- 仅使用图片阶段 screen-valid gaze；AOI 只用于空间归属。",
        "- 中心区主半径为 100 px，并检查 75/150 px；early escape 主窗为 0–500 ms，并检查 0–300/0–800 ms。",
        "- `residual_spatial_error` 是混合成分期望坐标与实际 gaze 的派生位置误差；mean NLL 是模型密度误差敏感性，两者不混写。",
        "- 主检验为 subject-level Spearman；六项 direct 指标在同一检验族内做 BH-FDR，CI 使用 2,000 次固定种子 bootstrap。",
        "",
        "## Direct correspondence 结果",
        "",
        "| direct 指标 | rho | BH-FDR | 95% CI | 去极端值 rho | 质量调整 rho | n_valid | 稳定性 | 等级 |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in direct.itertuples(index=False):
        lines.append(f"| `{row.direct_metric}` | {row.rho:.3f} | {format_p(row.p_bh_fdr)} | [{row.rho_ci95_low:.3f}, {row.rho_ci95_high:.3f}] | {row.rho_excluding_mad_extremes:.3f} | {row.rho_quality_adjusted_rank_partial:.3f} | {row.n_valid} | {row.metric_stability_level} | {row.interpretation_level} |")
    lines.extend([
        "",
        "## 参数补偿、拟合与质量诊断",
        "",
        f"- `dc` 奇偶 split-half：rho={split_row.rho:.3f}，95% CI [{split_row.ci95_low:.3f}, {split_row.ci95_high:.3f}]。",
        f"- 近似等似然解内达到 |rho|≥0.30 且 FDR<0.05 的补偿关系：{len(trade_signal)} 项。跨被试相关不单独作为参数吸收证据。",
        f"- 固定 cohort-median `dc` 后，在成功拟合中 AIC 更低者 {fixed_preferred}/{len(successful_fit)}，自由 `dc` 更低者 {free_preferred}/{len(successful_fit)}；固定拟合成功 {len(successful_fit)}/80。",
        f"- 与 `dc` 达到 FDR<0.05 的数据质量指标：{len(quality_signal)}/{int(quality['p_value'].notna().sum())} 个可估计关系；`valid_trial_count` 因所有人均为 40 而不可估计。MAD 极端值 {int(frame['dc_extreme_mad'].sum())}/80，拟合边界值 {int(frame['dc_boundary_flag'].sum())}/80。",
        "",
        "### 参数相关明细",
        "",
        "| 对象 | rho | BH-FDR | 解释 |",
        "| --- | ---: | ---: | --- |",
    ])
    for row in compensation[compensation["analysis_scope"] == "across_subject"].itertuples(index=False):
        interpretation = "相关线索，不能单独证明补偿" if row.y_variable.startswith(("pm_", "Tmax")) else "拟合质量关系"
        lines.append(f"| `{row.y_variable}` | {row.rho:.3f} | {format_p(row.p_bh_fdr)} | {interpretation} |")
    lines.extend([
        "",
        f"近似等似然解内唯一达到预设补偿门槛的是：{', '.join(trade_signal['y_variable'].tolist()) if not trade_signal.empty else '无'}。",
        "",
        "## 五层最终判断",
        "",
        "### 范式层",
        "快速三图任务已有 R01/R02 空间行为结构证据；R03c 不把该层证据自动转写成 `dc` 有效。",
        "",
        "### 指标层",
        f"新增六项 direct 空间指标中 {stable_metrics}/6 达到预设 split-half stable 门槛，其余指标按 cautious/unstable 降级。",
        "",
        "### 参数层",
        f"专项主检验中 `dc` 获得 strong={len(strong)}、cautious={len(cautious)}、unsupported={len(direct)-len(strong)-len(cautious)}。去除 MAD 极端值后仍显著正向 {len(extreme_robust)}/6 项，质量调整后仍显著正向 {len(quality_robust)}/6 项。当前最接近且最稳健的行为对象是 `{strongest.direct_metric}`；这区分了“旧指标不够贴合”和“参数仍缺乏对应”两种解释。",
        "",
        "### 稳定性/补偿层",
        f"`dc` split-half rho={split_row.rho:.3f}；近似等似然补偿信号 {len(trade_signal)} 项；质量依赖 {len(quality_signal)} 项。尺度混存已校正，但仍作为参数来源风险保留。",
        "",
        "### 后续建议",
        recommendation,
        "",
        "## 明确回答",
        "",
        f"1. 旧指标是否不够贴合：专项指标中有 {len(strong)+len(cautious)}/6 项至少达到 cautious；据此分层判断，不能只凭旧 R03 弱证据认定 `dc` 无对应。",
        f"2. `dc` 最接近的空间过程：`{strongest.direct_metric}`，具体强度见主表，其他过程不作等价解释。",
        f"3. 是否被 `pm`/`Tmax` 吸收：近似等似然解内有 {len(trade_signal)} 项预设补偿信号；跨被试相关仅作线索。",
        f"4. 是否受质量、极端值或边界影响：可估计质量关系中 FDR 信号 {len(quality_signal)}/{int(quality['p_value'].notna().sum())} 项、MAD 极端值 {int(frame['dc_extreme_mad'].sum())} 人、边界值 {int(frame['dc_boundary_flag'].sum())} 人；主表同时报告去极端值和质量调整后的相关。",
        f"5. 当前处置：{recommendation}",
        "",
        "## 非诊断边界",
        "",
        "本结果是正常样本中的范式内行为对应与拟合诊断，不构成抑郁症诊断、病理机制证明或个体临床准入依据。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_CSV, dtype={"subject_id": str})
    params = pd.read_csv(PARAM_CSV, dtype={"subject_id": str})
    repro = pd.read_csv(REPRO_CSV, dtype={"subject_id": str})
    best = pd.read_csv(BEST_CSV, dtype={"subject_id": str})
    runs = pd.read_csv(RUNS_CSV, dtype={"subject_id": str})
    split = pd.read_csv(SPLIT_CSV, dtype={"subject_id": str})
    scale = audit_dc_scale(params, repro)
    scale.to_csv(OUTPUT_DIR / "dc_scale_audit_N80.csv", index=False, encoding="utf-8-sig")

    fit_columns = [
        "production_nll", "production_mean_nll", "free_multistart_nll", "free_multistart_mean_nll",
        "fixed_dc_px", "fixed_dc_success", "fixed_dc_message", "fixed_dc_nll", "fixed_dc_mean_nll",
        "delta_nll_fixed_minus_free", "aic_free_dc", "aic_fixed_dc", "delta_aic_fixed_minus_free",
        "model_n_observations",
    ]
    cached_fit = None
    if all(column in metrics.columns for column in fit_columns):
        cached_fit = metrics[["subject_id", *fit_columns]].copy()
    derived_columns = set(scale.columns) - {"subject_id"}
    derived_columns.update({
        "Pmax_positive_多图", "Pmax_negative_多图", "Pmax_neutral_多图",
        "Tmax1", "Tmax2", "Tmax3", "pm_positive", "pm_negative", "pm_neutral",
        "dc_multistart_best_px", "dc_extreme_mad", "dc_boundary_flag", *fit_columns,
    })
    metrics = metrics.drop(columns=[column for column in derived_columns if column in metrics.columns])

    pmax_sum = params[["Pmax_positive_多图", "Pmax_negative_多图", "Pmax_neutral_多图"]].sum(axis=1)
    for emotion in ("positive", "negative", "neutral"):
        params[f"pm_{emotion}"] = params[f"Pmax_{emotion}_多图"] / pmax_sum
    params = params.rename(columns={
        "Tmax_positive_多图": "Tmax1", "Tmax_negative_多图": "Tmax2", "Tmax_neutral_多图": "Tmax3"
    })
    best_full = best[best["split"] == "full"].sort_values("best_log_likelihood", ascending=False).drop_duplicates("subject_id")
    frame = metrics.merge(scale, on="subject_id", how="inner").merge(
        params[["subject_id", "Pmax_positive_多图", "Pmax_negative_多图", "Pmax_neutral_多图", "Tmax1", "Tmax2", "Tmax3", "pm_positive", "pm_negative", "pm_neutral"]],
        on="subject_id", how="inner",
    ).merge(
        best_full[["subject_id", "dc"]].rename(columns={"dc": "dc_multistart_best_px"}), on="subject_id", how="inner"
    )
    fixed_dc = float(frame["dc_harmonized_px"].median())
    print(f"stage=fit_diagnostics cached={cached_fit is not None}", flush=True)
    if cached_fit is not None and not cached_fit["fixed_dc_success"].astype(bool).all():
        failed_ids = set(cached_fit.loc[~cached_fit["fixed_dc_success"].astype(bool), "subject_id"])
        retried = run_fit_diagnostics(frame[frame["subject_id"].isin(failed_ids)], best, fixed_dc)
        fit = pd.concat([cached_fit[~cached_fit["subject_id"].isin(failed_ids)], retried], ignore_index=True)
    else:
        fit = cached_fit if cached_fit is not None else run_fit_diagnostics(frame, best, fixed_dc)
    frame = frame.merge(fit, on="subject_id", how="inner")
    frame["dc_extreme_mad"] = mad_flags(frame["dc_harmonized_px"])
    tolerance = (DC_BOUNDS[1] - DC_BOUNDS[0]) * 0.001
    frame["dc_boundary_flag"] = (frame["dc_harmonized_px"] <= DC_BOUNDS[0] + tolerance) | (frame["dc_harmonized_px"] >= DC_BOUNDS[1] - tolerance)
    frame.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")

    print("stage=statistics", flush=True)
    reliability = metric_reliability(frame)
    direct = direct_correlations(frame, reliability)
    compensation = compensation_diagnostics(frame, runs, split)
    quality = correlation_rows(frame, QUALITY_METRICS, "data_quality")
    direct.to_csv(OUTPUT_DIR / "dc_specific_correlations_N80.csv", index=False, encoding="utf-8-sig")
    compensation.to_csv(OUTPUT_DIR / "dc_parameter_compensation_correlations_N80.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUTPUT_DIR / "dc_quality_diagnostics_N80.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(OUTPUT_DIR / "dc_specific_metric_reliability_N80.csv", index=False, encoding="utf-8-sig")
    print("stage=figures", flush=True)
    make_figures(frame, direct, reliability, quality)
    print("stage=report", flush=True)
    write_report(frame, scale, direct, reliability, compensation, quality, fit)
    print(f"subjects={frame['subject_id'].nunique()} fixed_dc={fixed_dc:.6f}")
    print(direct[["direct_metric", "rho", "p_bh_fdr", "interpretation_level"]].to_string(index=False))
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
