#!/usr/bin/env python3
"""Freeze the formal T6 recovery design without running recovery outcomes."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parameter_recovery_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_text,
    robust_maximin_select,
)


TASK_ID = "T20260725-007-field-common-recovery-rules"
REPO_ROOT = SCRIPT_DIR.parents[4]
INPUT_ROOT = REPO_ROOT / "t5t6 联合数据分析" / "260616" / "data" / "20260722_t6_118_binocular_rerun"
PARAMETER_PATH = INPUT_ROOT / "t6_results.csv"
MANIFEST_PATH = INPUT_ROOT / "input_manifest.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "T6_v1" / "formal_recovery_design_v2_field_common"
SCREEN_WIDTH = 1920.0
SCREEN_HEIGHT = 1080.0
TRUTH_COUNT = 16
REPEATS_PER_TRUTH = 100
BLIND_STARTS = 5
RANDOM_SEED = 20260725
FIT_BOUNDS = {
    "Tmax1": [1.0, 3500.0], "Tmax2": [1.0, 3500.0], "Tmax3": [1.0, 3500.0],
    "pm_logit_positive": [-6.0, 6.0], "pm_logit_negative": [-6.0, 6.0],
    "dc_degree": [0.1, 10.0],
}
PARAMETERS = (
    "Tmax1", "Tmax2", "Tmax3",
    "pm_positive", "pm_negative", "pm_neutral", "dc",
)
SOURCE_COLUMNS = {
    "Pm1_positive": "pm_positive",
    "Pm2_negative": "pm_negative",
    "Pm3_neutral": "pm_neutral",
    "Tmax1": "Tmax1",
    "Tmax2": "Tmax2",
    "Tmax3": "Tmax3",
    "dc": "dc",
}
TRANSFORMED_SELECTION_COLUMNS = (
    "log_Tmax1", "log_Tmax2", "log_Tmax3",
    "alr_pm_positive", "alr_pm_negative", "log_dc",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_pair(value: Any) -> tuple[float, float]:
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, tuple) or len(parsed) != 2:
        raise ValueError(f"Invalid coordinate/size pair: {value}")
    return float(parsed[0]), float(parsed[1])


def normalize_emotion(value: Any) -> str:
    text = str(value).strip().lower()
    return {"postive": "positive", "positve": "positive"}.get(text, text)


def audit_records(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"records_readable": False, "trial_count": 0, "layout_valid": False, "emotion_complete": False}
    frame = pd.read_csv(path)
    position_columns = ("左图片坐标", "中图片坐标", "右图片坐标")
    size_columns = ("左图片大小", "中图片大小", "右图片大小")
    emotion_columns = ("左情绪类型", "中情绪类型", "右情绪类型")
    timing_columns = ("刺激图片开始时间", "刺激图片结束时间")
    required = set(position_columns + size_columns + emotion_columns + timing_columns)
    if not required.issubset(frame.columns):
        return {"records_readable": True, "trial_count": len(frame), "layout_valid": False, "emotion_complete": False}
    layout_valid = True
    unique_rectangles: set[tuple[float, float, float, float]] = set()
    for _, row in frame.iterrows():
        rectangles = []
        for position_column, size_column in zip(position_columns, size_columns):
            x, y = parse_pair(row[position_column])
            width, height = parse_pair(row[size_column])
            rectangle = (x, x + width, y, y + height)
            rectangles.append(rectangle)
            unique_rectangles.add(rectangle)
            if x < 0 or y < 0 or x + width > SCREEN_WIDTH or y + height > SCREEN_HEIGHT or width <= 0 or height <= 0:
                layout_valid = False
        for left_index in range(3):
            for right_index in range(left_index + 1, 3):
                left, right = rectangles[left_index], rectangles[right_index]
                overlap_x = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
                overlap_y = max(0.0, min(left[3], right[3]) - max(left[2], right[2]))
                if overlap_x * overlap_y > 0:
                    layout_valid = False
    emotion_complete = bool(frame.apply(
        lambda row: {normalize_emotion(row[column]) for column in emotion_columns}
        == {"positive", "negative", "neutral"}, axis=1,
    ).all())
    durations = pd.to_numeric(frame[timing_columns[1]], errors="coerce") - pd.to_numeric(frame[timing_columns[0]], errors="coerce")
    return {
        "records_readable": True,
        "trial_count": int(len(frame)),
        "layout_valid": bool(layout_valid),
        "emotion_complete": emotion_complete,
        "picture_duration_min_ms": float(durations.min()),
        "picture_duration_median_ms": float(durations.median()),
        "picture_duration_max_ms": float(durations.max()),
        "unique_rectangles": int(len(unique_rectangles)),
    }


def load_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    parameters = pd.read_csv(PARAMETER_PATH, dtype={"subject_id": str})
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"subject_id": str})
    if len(parameters) != 118 or parameters["subject_id"].nunique() != 118:
        raise ValueError("Current T6 parameter table must contain 118 unique subjects")
    if len(manifest) != 118 or manifest["subject_id"].nunique() != 118:
        raise ValueError("Current T6 manifest must contain 118 unique subjects")
    parameters = parameters[["subject_id", *SOURCE_COLUMNS]].rename(columns=SOURCE_COLUMNS)
    merged = parameters.merge(
        manifest[["subject_id", "eyetrack_path", "records_path", "status"]],
        on="subject_id", how="inner", validate="one_to_one",
    )
    numeric = merged[list(PARAMETERS)].apply(pd.to_numeric, errors="coerce")
    merged[list(PARAMETERS)] = numeric
    pm_sum = merged[["pm_positive", "pm_negative", "pm_neutral"]].sum(axis=1)
    merged["parameter_valid"] = (
        np.isfinite(numeric).all(axis=1)
        & (merged[["Tmax1", "Tmax2", "Tmax3", "pm_positive", "pm_negative", "pm_neutral", "dc"]] > 0).all(axis=1)
        & np.isclose(pm_sum, 1.0, atol=1e-8)
    )
    audits = []
    for row in merged.itertuples(index=False):
        audit = audit_records(Path(str(row.records_path)))
        audits.append({"subject_id": row.subject_id, **audit})
    audit_frame = pd.DataFrame(audits)
    merged = merged.merge(audit_frame, on="subject_id", how="left", validate="one_to_one")
    merged["template_eligible"] = (
        merged["parameter_valid"]
        & merged["status"].eq("success")
        & merged["records_readable"]
        & merged["trial_count"].eq(40)
        & merged["layout_valid"]
        & merged["emotion_complete"]
        & merged["eyetrack_path"].map(lambda value: Path(str(value)).is_file())
    )
    eligible = merged.loc[merged["template_eligible"]].copy()
    if len(eligible) < TRUTH_COUNT:
        raise ValueError(f"Need at least {TRUTH_COUNT} eligible T6 templates, found {len(eligible)}")
    for name in ("Tmax1", "Tmax2", "Tmax3", "dc"):
        eligible[f"log_{name}"] = np.log(eligible[name])
    eligible["alr_pm_positive"] = np.log(eligible["pm_positive"] / eligible["pm_neutral"])
    eligible["alr_pm_negative"] = np.log(eligible["pm_negative"] / eligible["pm_neutral"])
    return merged, eligible


def empirical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter in PARAMETERS:
        values = frame[parameter].astype(float)
        quantiles = values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        rows.append({
            "parameter": parameter,
            "unit": "ms" if parameter.startswith("Tmax") else ("degree" if parameter == "dc" else "proportion"),
            "n": len(values), "min": values.min(), "p01": quantiles.loc[0.01],
            "p05": quantiles.loc[0.05], "p25": quantiles.loc[0.25], "median": quantiles.loc[0.50],
            "p75": quantiles.loc[0.75], "p95": quantiles.loc[0.95], "p99": quantiles.loc[0.99],
            "max": values.max(), "mean": values.mean(), "sd": values.std(ddof=1),
            "p05_p95_span": quantiles.loc[0.95] - quantiles.loc[0.05],
        })
    return pd.DataFrame(rows)


def select_truths(eligible: pd.DataFrame) -> pd.DataFrame:
    selected = robust_maximin_select(
        eligible, list(TRANSFORMED_SELECTION_COLUMNS), TRUTH_COUNT, id_column="subject_id",
    )
    output_columns = [
        "truth_id", "selection_order", "subject_id", *PARAMETERS,
        "eyetrack_path", "records_path", "trial_count", "picture_duration_min_ms",
        "picture_duration_median_ms", "picture_duration_max_ms", "unique_rectangles",
    ]
    return selected[output_columns].copy()


def start_bank(summary: pd.DataFrame) -> pd.DataFrame:
    stats = summary.set_index("parameter")
    median_dc = float(stats.loc["dc", "median"])
    low_dc = float(stats.loc["dc", "p25"])
    high_dc = float(stats.loc["dc", "p75"])
    low_t = [float(stats.loc[name, "p25"]) for name in ("Tmax1", "Tmax2", "Tmax3")]
    mid_t = [float(stats.loc[name, "median"]) for name in ("Tmax1", "Tmax2", "Tmax3")]
    high_t = [float(stats.loc[name, "p75"]) for name in ("Tmax1", "Tmax2", "Tmax3")]
    specifications = (
        ("empirical_median", mid_t, [1 / 3, 1 / 3, 1 / 3], median_dc),
        ("early_narrow", low_t, [1 / 3, 1 / 3, 1 / 3], low_dc),
        ("late_wide", high_t, [1 / 3, 1 / 3, 1 / 3], high_dc),
        ("staggered", [low_t[0], mid_t[1], high_t[2]], [0.50, 0.30, 0.20], median_dc),
        ("reverse_staggered", [high_t[0], mid_t[1], low_t[2]], [0.20, 0.50, 0.30], median_dc),
    )
    return pd.DataFrame([
        {
            "start_order": index, "start_id": name,
            "Tmax1": tmax[0], "Tmax2": tmax[1], "Tmax3": tmax[2],
            "pm_positive": pm[0], "pm_negative": pm[1], "pm_neutral": pm[2], "dc": dc,
        }
        for index, (name, tmax, pm, dc) in enumerate(specifications, start=1)
    ])


def decision_rules() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "rule_id": "implementation_contract", "scope": "all_datasets",
            "metric": "reproducible_generation_and_finite_density",
            "primary_rule": "100% required",
            "basis": "logical implementation requirement; failure invalidates recovery inference",
        },
        {
            "rule_id": "optimizer_completion", "scope": "each_parameter_family",
            "metric": "success_failure_and_boundary_fractions_with_intervals",
            "primary_rule": "mandatory continuous report; no universal numerical cutoff",
            "basis": "optimizer behavior is part of the recovery evidence and must be disclosed",
        },
        {
            "rule_id": "rank_recovery", "scope": "each_reported_parameter",
            "metric": "truth_vs_recovered_Spearman_rho_with_95pct_interval_across_truth_level_medians",
            "primary_rule": "mandatory continuous estimate and plot; no universal numerical cutoff",
            "basis": "Wilson and Collins (2019): inspect truth-recovery relation over empirically relevant ranges",
        },
        {
            "rule_id": "estimation_error", "scope": "each_reported_parameter",
            "metric": "bias_MAE_RMSE_median_absolute_error_and_MCSE_in_original_and_normalized_units",
            "primary_rule": "mandatory continuous report with uncertainty; no universal numerical cutoff",
            "basis": "Morris et al. (2019): report multiple performance measures and Monte Carlo uncertainty",
        },
        {
            "rule_id": "range_dependence", "scope": "each_truth_and_parameter",
            "metric": "performance_stratified_by_truth_profile_and_empirical_range",
            "primary_rule": "mandatory report",
            "basis": "recovery can vary across the parameter space; aggregate summaries must not conceal local weakness",
        },
        {
            "rule_id": "parameter_confusion", "scope": "all_parameters",
            "metric": "truth_to_recovered_cross_parameter_correlation_matrix",
            "primary_rule": "mandatory report; no post-hoc deletion",
            "basis": "Wilson and Collins (2019) recommend checking induced parameter correlations/trade-offs",
        },
        {
            "rule_id": "evidence_conclusion", "scope": "Tmax_pm_dc",
            "metric": "joint_interpretation_of_all_prespecified_performance_measures",
            "primary_rule": "describe evidence strength and limitations; do not collapse to a universal binary pass line",
            "basis": "field guidance supports transparent multimetric evaluation rather than a universal cutoff",
        },
    ])


def make_figures(all_rows: pd.DataFrame, truths: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.2))
    for ax, parameter in zip(axes.ravel(), PARAMETERS):
        ax.hist(all_rows[parameter], bins=18, color="#D9E3F0", edgecolor="white", label="N118")
        for value in truths[parameter]:
            ax.axvline(value, color="#E45756", alpha=0.45, linewidth=0.8)
        ax.set_title(parameter)
        ax.set_ylabel("Count")
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(frameon=False)
    fig.suptitle("T6 formal truth-bank coverage of the empirical parameter range")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"figure-01-truth-bank-coverage.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)

    summary = empirical_summary(all_rows).set_index("parameter")
    rows = []
    for parameter in PARAMETERS:
        span = float(summary.loc[parameter, "p05_p95_span"])
        center = float(summary.loc[parameter, "median"])
        rows.append((parameter, (truths[parameter].min() - center) / span, (truths[parameter].max() - center) / span))
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    y = np.arange(len(rows))
    lower = np.array([item[1] for item in rows])
    upper = np.array([item[2] for item in rows])
    ax.hlines(y, lower, upper, color="#4C78A8", linewidth=4)
    ax.scatter(lower, y, color="#4C78A8")
    ax.scatter(upper, y, color="#4C78A8")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, [item[0] for item in rows])
    ax.set_xlabel("Selected min/max relative to N118 median (P05-P95 span units)")
    ax.set_title("Multivariate truth-bank range audit")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"figure-02-truth-range-audit.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def render_reports(
    all_rows: pd.DataFrame, eligible: pd.DataFrame, truths: pd.DataFrame,
    summary: pd.DataFrame, rules: pd.DataFrame, contract: dict[str, Any], output_dir: Path,
) -> None:
    mcse_worst = math.sqrt(0.25 / REPEATS_PER_TRUTH)
    mcse_80 = math.sqrt(0.8 * 0.2 / REPEATS_PER_TRUTH)
    report = f"""# T6正式参数恢复预注册设计

## 当前结论

领域通用版正式设计已冻结，但正式模拟尚未运行。该顺序保证真值范围、重复次数、拟合起点、指标和报告规则不会根据正式结果事后调整。

## ADEMP设计

- Aim：检验T6任务在当前40-trial结构下能否恢复Tmax、pm和dc。
- Data-generating mechanism：使用已通过排错的规范化观测模型，并逐真值复用对应被试的实际40-trial时间表、三图坐标和图片尺寸。
- Estimands：Tmax1/2/3、pm-positive/negative/neutral和dc。
- Methods：每个模拟数据集使用5个真值盲、确定性起点拟合，按最高有效似然选择结果；生成与回拟合使用同一观测密度。
- Performance measures：偏差、MAE、RMSE、中位绝对误差、Spearman相关及不确定区间、失败率、边界率和真值—恢复值参数混淆矩阵。

## 真值库

- 当前双眼参数：{len(all_rows)}人。
- 具有有效参数及可读取40-trial真实模板：{len(eligible)}人。
- 正式真值：{len(truths)}组。
- 选择方法：在log-Tmax、两项pm对数比和log-dc构成的6维空间做确定性robust maximin选择。
- 依据：Wilson与Collins建议在已有真实拟合时，让模拟范围匹配真实参数范围；maximin用于覆盖多变量空间，而不是挑选结果最好的人。

## 模拟规模

- 每组真值独立重复：{REPEATS_PER_TRUTH}次。
- 正式数据集：{len(truths) * REPEATS_PER_TRUTH}个。
- 盲拟合：{len(truths) * REPEATS_PER_TRUTH * BLIND_STARTS}次。
- 最坏情况下二项比例Monte Carlo SE：{mcse_worst:.3f}；比例为0.80时MCSE：{mcse_80:.3f}。
- 依据：重复数按Monte Carlo误差明确，而不是沿用旧pilot的便利数量；同时保留T5的16组maximin真值与5个盲起点结构。

## 预先固定的拟合范围

- Tmax：1–3500 ms；覆盖当前N118的72.7–2105.2 ms，并超过实际最长图片页约3019 ms。
- pm：通过两个logit估计并自动闭合为三项和等于1；logit范围为−6至6。
- dc：0.1–10.0°；覆盖当前N118的0.349–7.734°。
- 这些范围只用于本次规范化恢复验证，已在查看正式结果前固定；不得根据正式结果无版本地放宽或收紧。

## 统计单位

- 排序恢复：先对每组真值的100次恢复取中位数，再在16组真值之间计算Spearman相关，避免把重复模拟误当成新的真值样本。
- 误差与失败率：以模拟数据集为单位，同时按真值组分层汇总。
- 不确定性：报告Monte Carlo SE；偏差和误差使用按真值分层的bootstrap区间；失败率报告精确或Wilson区间。
- pm：除分量误差外，补充组成数据距离；三个分量不得被当作三个相互独立的自由变量。

## 判断规则的证据边界

本设计不使用项目自定的相关或偏差通过线。正式判断依据预先声明的完整证据组合：实现正确性、优化失败与边界比例、真值—恢复值图及相关区间、偏差与误差及其Monte Carlo不确定性、不同真值范围内的表现，以及参数混淆。每个参数和参数族按实际证据强弱与用途限制进行描述，不把连续结果压缩成领域内不存在的统一二元门槛。

## 外部依据

- Wilson & Collins (2019), *eLife*, DOI: 10.7554/eLife.49547：参数范围应匹配真实拟合范围，并应绘制真值—恢复值、检查偏差和参数间混淆。
- Morris, White & Crowther (2019), *Statistics in Medicine*, DOI: 10.1002/sim.8086：模拟研究应预先定义ADEMP结构，并报告Monte Carlo误差。
- 项目定向同领域核查：参数恢复在方法型论文中常见，但真实数据逐被试profile不是合成恢复通过的常规硬门槛。

## 冻结状态

`DESIGN_V2_FIELD_COMMON_READY_FOR_RUN`

任何正式运行后的设计变更必须新建版本，并同时保留本版本及变更理由。
"""
    appendix = f"""# T6正式参数恢复设计统计附录

## N118参数范围

```csv
{summary.to_csv(index=False).strip()}
```

## 预声明判断规则

```csv
{rules.to_csv(index=False).strip()}
```

## Monte Carlo精度

- repeats per truth = {REPEATS_PER_TRUTH}
- maximum binomial MCSE = {mcse_worst:.6f}
- binomial MCSE at p=0.80 = {mcse_80:.6f}
- 正式报告还需为每项偏差、MAE和RMSE给出相应MCSE或分层bootstrap区间。

## 尚未产生的证据

- 正式恢复值、正式相关和误差均尚未产生。
- 当前文件不得用于声称T6参数已经正式通过恢复。
"""
    catalog = """# Figure catalog

## Figure 1: truth-bank coverage

- 文件：`figures/figure-01-truth-bank-coverage.png` / `.pdf`
- 目的：检查16组正式真值是否覆盖当前N118参数分布。
- 数据：当前双眼N118参数与冻结真值库。
- 读图：浅色直方图为N118；红线为每组正式真值。
- 决策：若真值集中在分布局部，则不能启动正式恢复。
- 限制：该图只评价设计覆盖，不是恢复结果。

## Figure 2: multivariate range audit

- 文件：`figures/figure-02-truth-range-audit.png` / `.pdf`
- 目的：按N118 P05-P95跨度统一检查各参数的真值覆盖宽度。
- 数据：当前双眼N118参数与冻结真值库。
- 读图：每条线表示所选真值相对经验中位数的最小—最大范围。
- 决策：发现某参数覆盖不足时必须在正式运行前处理。
- 限制：不同参数仍保留各自科学单位；标准化仅用于设计检查。
"""
    atomic_write_text(report, output_dir / "analysis-report.md")
    atomic_write_text(appendix, output_dir / "stats-appendix.md")
    atomic_write_text(catalog, output_dir / "figure-catalog.md")
    atomic_write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True), output_dir / "formal-design-contract.json")


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, eligible = load_candidates()
    summary = empirical_summary(all_rows)
    truths = select_truths(eligible)
    starts = start_bank(summary)
    rules = decision_rules()
    mcse_worst = math.sqrt(0.25 / REPEATS_PER_TRUTH)
    contract = {
        "task_id": TASK_ID,
        "status": "DESIGN_V2_FIELD_COMMON_READY_FOR_RUN",
        "design_version": 2,
        "design_framework": "ADEMP",
        "style_reference": "T5_truth_bank_matched_generation_blind_multistart_recovery_metrics",
        "input_parameter_sha256": sha256_file(PARAMETER_PATH),
        "input_manifest_sha256": sha256_file(MANIFEST_PATH),
        "designer_sha256": sha256_file(Path(__file__)),
        "random_seed": RANDOM_SEED,
        "truth_count": TRUTH_COUNT,
        "truth_selection": "robust_maximin_on_log_Tmax_ALR_pm_log_dc",
        "repeats_per_truth": REPEATS_PER_TRUTH,
        "formal_datasets": TRUTH_COUNT * REPEATS_PER_TRUTH,
        "blind_starts": BLIND_STARTS,
        "formal_fits": TRUTH_COUNT * REPEATS_PER_TRUTH * BLIND_STARTS,
        "trials_per_dataset": 40,
        "fit_bounds": FIT_BOUNDS,
        "template_rule": "truth_subject_actual_formal_records_and_eyetrack_time_grid",
        "generation_refit_rule": "identical_normalized_observation_density",
        "selection_unit": "truth_level_median_across_repeats",
        "error_unit": "simulation_dataset_stratified_by_truth",
        "maximum_binomial_mcse": mcse_worst,
        "formal_outcomes_generated": False,
        "references": ["10.7554/eLife.49547", "10.1002/sim.8086"],
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    contract["contract_hash"] = hashlib.sha256(canonical).hexdigest()
    bounds_ok = bool(
        truths[["Tmax1", "Tmax2", "Tmax3"]].ge(FIT_BOUNDS["Tmax1"][0]).all().all()
        and truths[["Tmax1", "Tmax2", "Tmax3"]].le(FIT_BOUNDS["Tmax1"][1]).all().all()
        and truths["dc"].between(*FIT_BOUNDS["dc_degree"]).all()
    )
    readiness = pd.DataFrame([{
        "task_id": TASK_ID,
        "n118_parameters": len(all_rows),
        "eligible_actual_templates": len(eligible),
        "truths_selected": len(truths),
        "all_truth_templates_40_trials": bool(truths["trial_count"].eq(40).all()),
        "all_truths_inside_frozen_bounds": bounds_ok,
        "five_unique_blind_starts": bool(len(starts) == BLIND_STARTS and starts["start_id"].nunique() == BLIND_STARTS),
        "formal_outcomes_generated": False,
        "design_status": "DESIGN_V2_FIELD_COMMON_READY_FOR_RUN",
    }])
    atomic_write_csv(truths, output_dir / "formal-truth-bank.csv")
    atomic_write_csv(starts, output_dir / "blind-start-bank.csv")
    atomic_write_csv(summary, output_dir / "empirical-parameter-summary.csv")
    atomic_write_csv(rules, output_dir / "decision-rules.csv")
    atomic_write_csv(readiness, output_dir / "design-readiness.csv")
    make_figures(all_rows, truths, output_dir)
    render_reports(all_rows, eligible, truths, summary, rules, contract, output_dir)
    return {
        "output_dir": str(output_dir), "status": contract["status"],
        "eligible": len(eligible), "truths": len(truths),
        "datasets_planned": contract["formal_datasets"], "fits_planned": contract["formal_fits"],
    }


if __name__ == "__main__":
    print(run())
