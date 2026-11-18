#!/usr/bin/env python3
"""Re-summarize existing T5 recovery and define shared T5/T6 field-common reporting."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
T5_ROOT = SCRIPT_DIR / "outputs" / "T5"
T5_FULL = T5_ROOT / "full"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "field_common_recovery_framework"
MAIN_CONDITION = "main_24_trial"
BOOTSTRAP_REPEATS = 2000
RANDOM_SEED = 20260725
PARAMETERS = (
    "positive_tau", "positive_sigma_s", "positive_d_c",
    "negative_tau", "negative_sigma_s", "negative_d_c",
    "neutral_tau", "neutral_sigma_s", "neutral_d_c",
    "complex_tau", "complex_sigma_s", "complex_d_c",
)
PARAMETER_UNITS = {
    **{name: "model_time_unit" for name in PARAMETERS if name.endswith("_tau")},
    **{name: "pixel" for name in PARAMETERS if name.endswith("_sigma_s") or name.endswith("_d_c")},
}
BOUNDS_BY_KIND = {"tau": (0.01, 10.0), "sigma_s": (5.0, 600.0), "d_c": (5.0, 400.0)}
BOUNDARY_FRACTION = 0.001


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    left = pd.Series(x, dtype=float).rank(method="average")
    right = pd.Series(y, dtype=float).rank(method="average")
    return float(left.corr(right))


def load_t5() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truths = pd.read_csv(T5_ROOT / "truth_parameter_bank.csv")
    runs = pd.read_csv(T5_FULL / "recovery_runs.csv", low_memory=False)
    manifest = pd.read_csv(T5_FULL / "simulation_manifest.csv")
    selected = runs.loc[
        runs["condition"].eq(MAIN_CONDITION)
        & runs["fit_role"].eq("formal")
        & runs["is_best_formal"].astype(str).str.lower().eq("true")
    ].copy()
    if len(truths) != 16 or truths["truth_id"].nunique() != 16:
        raise ValueError("T5 truth bank must contain 16 unique truths")
    expected = manifest.loc[manifest["condition"].eq(MAIN_CONDITION)]
    if len(expected) != 960 or expected["run_id"].nunique() != 960:
        raise ValueError("T5 main condition must contain 960 simulation datasets")
    if len(selected) != 960 or selected["run_id"].nunique() != 960:
        raise ValueError("Expected one best formal fit for each of 960 T5 main datasets")
    for index, parameter in enumerate(PARAMETERS):
        selected[parameter] = pd.to_numeric(selected[f"theta_{index}"], errors="coerce")
    if not np.isfinite(selected[list(PARAMETERS)].to_numpy(dtype=float)).all():
        raise ValueError("T5 selected recovery estimates contain non-finite values")
    return truths, selected, expected


def summarize_t5(
    truths: pd.DataFrame, recovered: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    truth_long = truths[["truth_id", *PARAMETERS]].melt(
        id_vars="truth_id", var_name="parameter", value_name="truth",
    )
    recovered_long = recovered[["run_id", "truth_id", "sim", "success", *PARAMETERS]].melt(
        id_vars=["run_id", "truth_id", "sim", "success"],
        value_vars=PARAMETERS, var_name="parameter", value_name="recovered",
    ).merge(truth_long, on=["truth_id", "parameter"], how="left", validate="many_to_one")
    recovered_long["error"] = recovered_long["recovered"] - recovered_long["truth"]
    recovered_long["absolute_error"] = recovered_long["error"].abs()
    spans = truths[list(PARAMETERS)].max() - truths[list(PARAMETERS)].min()
    recovered_long["truth_range_span"] = recovered_long["parameter"].map(spans)
    recovered_long["normalized_error"] = recovered_long["error"] / recovered_long["truth_range_span"]

    per_truth_rows = []
    for (truth_id, parameter), group in recovered_long.groupby(["truth_id", "parameter"], sort=True):
        errors = group["error"].to_numpy(dtype=float)
        per_truth_rows.append({
            "truth_id": truth_id,
            "parameter": parameter,
            "unit": PARAMETER_UNITS[parameter],
            "truth": float(group["truth"].iloc[0]),
            "n_repeats": len(group),
            "median_recovered": float(group["recovered"].median()),
            "mean_recovered": float(group["recovered"].mean()),
            "bias": float(errors.mean()),
            "mcse_bias": float(errors.std(ddof=1) / math.sqrt(len(errors))),
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "median_absolute_error": float(np.median(np.abs(errors))),
            "normalized_bias": float(group["normalized_error"].mean()),
            "normalized_mae": float(group["normalized_error"].abs().mean()),
        })
    per_truth = pd.DataFrame(per_truth_rows)

    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows = []
    for parameter in PARAMETERS:
        detail = recovered_long.loc[recovered_long["parameter"].eq(parameter)]
        truth_level = per_truth.loc[per_truth["parameter"].eq(parameter)].sort_values("truth_id")
        rho = spearman(truth_level["truth"], truth_level["median_recovered"])
        by_truth = {
            truth_id: detail.loc[detail["truth_id"].eq(truth_id), "recovered"].to_numpy(dtype=float)
            for truth_id in truth_level["truth_id"]
        }
        boot = np.empty(BOOTSTRAP_REPEATS, dtype=float)
        truth_values = truth_level["truth"].to_numpy(dtype=float)
        for index in range(BOOTSTRAP_REPEATS):
            medians = [float(np.median(rng.choice(by_truth[truth_id], size=len(by_truth[truth_id]), replace=True)))
                       for truth_id in truth_level["truth_id"]]
            boot[index] = spearman(truth_values, medians)
        errors = detail["error"].to_numpy(dtype=float)
        normalized = detail["normalized_error"].to_numpy(dtype=float)
        kind = next(kind for kind in BOUNDS_BY_KIND if parameter.endswith(kind))
        lower, upper = BOUNDS_BY_KIND[kind]
        tolerance = BOUNDARY_FRACTION * (upper - lower)
        boundary_fraction = float(((detail["recovered"] <= lower + tolerance)
                                   | (detail["recovered"] >= upper - tolerance)).mean())
        summary_rows.append({
            "parameter": parameter,
            "unit": PARAMETER_UNITS[parameter],
            "n_truths": truth_level["truth_id"].nunique(),
            "repeats_per_truth": int(detail.groupby("truth_id").size().min()),
            "n_datasets": len(detail),
            "truth_level_spearman_rho": rho,
            "rho_bootstrap_95pct_low": float(np.nanquantile(boot, 0.025)),
            "rho_bootstrap_95pct_high": float(np.nanquantile(boot, 0.975)),
            "bias": float(errors.mean()),
            "mcse_bias": float(errors.std(ddof=1) / math.sqrt(len(errors))),
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "median_absolute_error": float(np.median(np.abs(errors))),
            "normalized_bias": float(normalized.mean()),
            "normalized_mae": float(np.abs(normalized).mean()),
            "normalized_rmse": float(np.sqrt(np.mean(normalized ** 2))),
            "boundary_fraction": boundary_fraction,
        })
    summary = pd.DataFrame(summary_rows)

    recovered_wide = per_truth.pivot(index="truth_id", columns="parameter", values="median_recovered")
    truth_wide = truths.set_index("truth_id")[list(PARAMETERS)]
    truth_corr = truth_wide.corr(method="spearman")
    recovered_corr = recovered_wide.corr(method="spearman")
    cross_corr = pd.DataFrame(index=PARAMETERS, columns=PARAMETERS, dtype=float)
    for truth_parameter in PARAMETERS:
        for recovered_parameter in PARAMETERS:
            cross_corr.loc[truth_parameter, recovered_parameter] = spearman(
                truth_wide[truth_parameter], recovered_wide[recovered_parameter],
            )
    error_wide = recovered_long.pivot(index="run_id", columns="parameter", values="error")
    error_corr = error_wide.corr(method="spearman")
    matrices = {
        "truth-correlation-matrix": truth_corr,
        "recovered-median-correlation-matrix": recovered_corr,
        "truth-to-recovered-cross-correlation-matrix": cross_corr,
        "within-truth-error-correlation-matrix": error_corr,
    }

    implementation = pd.DataFrame([{
        "condition": MAIN_CONDITION,
        "truths": recovered["truth_id"].nunique(),
        "datasets": recovered["run_id"].nunique(),
        "repeats_per_truth_min": int(recovered.groupby("truth_id").size().min()),
        "repeats_per_truth_max": int(recovered.groupby("truth_id").size().max()),
        "finite_estimate_fraction": float(np.isfinite(recovered[list(PARAMETERS)]).all(axis=1).mean()),
        "selected_fit_success_fraction": float(recovered["success"].astype(str).str.lower().eq("true").mean()),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
    }])
    return summary, per_truth, matrices, implementation


def make_figures(per_truth: pd.DataFrame, matrices: dict[str, pd.DataFrame], output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.0))
    for ax, parameter in zip(axes.ravel(), PARAMETERS):
        frame = per_truth.loc[per_truth["parameter"].eq(parameter)].sort_values("truth")
        ax.scatter(frame["truth"], frame["median_recovered"], color="#4C78A8", s=24)
        limits = [min(frame["truth"].min(), frame["median_recovered"].min()),
                  max(frame["truth"].max(), frame["median_recovered"].max())]
        ax.plot(limits, limits, linestyle="--", linewidth=0.8, color="black")
        ax.set_title(parameter)
        ax.set_xlabel("True value")
        ax.set_ylabel("Median recovered value")
    fig.suptitle("T5 truth-level parameter recovery (main 24-trial condition)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"figure-01-t5-truth-vs-recovered.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)

    selected = (
        ("truth-correlation-matrix", "True parameters"),
        ("recovered-median-correlation-matrix", "Recovered medians"),
        ("within-truth-error-correlation-matrix", "Within-truth recovery errors"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(20.0, 7.2))
    image = None
    for ax, (key, title) in zip(axes, selected):
        matrix = matrices[key].loc[list(PARAMETERS), list(PARAMETERS)].to_numpy(dtype=float)
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(np.arange(len(PARAMETERS)), PARAMETERS, rotation=65, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(PARAMETERS)), PARAMETERS, fontsize=7)
        ax.set_title(title)
    assert image is not None
    colorbar_axis = fig.add_axes([0.955, 0.28, 0.012, 0.48])
    fig.colorbar(image, cax=colorbar_axis, label="Spearman correlation")
    fig.suptitle("T5 parameter dependence and recovery-error confusion")
    fig.subplots_adjust(left=0.07, right=0.935, bottom=0.25, top=0.86, wspace=0.32)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"figure-02-t5-parameter-confusion.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def shared_rules() -> pd.DataFrame:
    return pd.DataFrame([
        {"evidence": "implementation", "required_report": "reproducible generation, finite observations, fit failures and boundary cases", "universal_cutoff": "none; implementation failures are disclosed and investigated"},
        {"evidence": "truth_recovery_relation", "required_report": "truth-versus-recovered plots and truth-level rank association with interval", "universal_cutoff": "none"},
        {"evidence": "estimation_error", "required_report": "bias, MAE, RMSE, median absolute error and Monte Carlo uncertainty in scientific and normalized units", "universal_cutoff": "none"},
        {"evidence": "range_dependence", "required_report": "performance by truth profile and across the empirical parameter range", "universal_cutoff": "none"},
        {"evidence": "parameter_confusion", "required_report": "full truth-to-recovered cross-parameter association matrix", "universal_cutoff": "none"},
        {"evidence": "conclusion", "required_report": "joint evidence strength, limitations and intended-use implications for every parameter family", "universal_cutoff": "no single binary pass line"},
    ])


def render_reports(
    summary: pd.DataFrame, implementation: pd.DataFrame,
    matrices: dict[str, pd.DataFrame], output_dir: Path,
) -> None:
    rho_min = float(summary["truth_level_spearman_rho"].min())
    rho_max = float(summary["truth_level_spearman_rho"].max())
    error_corr = matrices["within-truth-error-correlation-matrix"].copy()
    np.fill_diagonal(error_corr.values, np.nan)
    max_position = np.unravel_index(np.nanargmax(np.abs(error_corr.values)), error_corr.shape)
    max_error_pair = (error_corr.index[max_position[0]], error_corr.columns[max_position[1]])
    max_error_correlation = float(error_corr.iloc[max_position])
    report = f"""# T5/T6参数恢复：领域通用报告框架

## 已完成

T5不重新生成模拟数据。本次复用既有16组真值、主条件每组60次、共960个模拟数据集，以真值组为统计单位重新汇总12个自由参数。已补充真值—恢复值图、真值层级Spearman相关及分层bootstrap区间、偏差的Monte Carlo标准误、连续误差指标、逐真值表现、逐参数边界率和参数混淆检查。

T6采用同一报告框架；正式设计已另存为领域通用版，正式恢复结果尚未运行。

## 统一判断方式

不设置领域内并不存在的统一相关或标准化偏差通过线。正式结论联合考虑：实现与优化状态、真值—恢复值关系、估计误差及Monte Carlo不确定性、参数空间中的局部表现和参数混淆。结果按证据强弱和实际用途逐参数解释，不压缩为单一二元标签。

## T5重新汇总的事实

- 主条件模拟数据集：{int(implementation.loc[0, 'datasets'])}。
- 真值组：{int(implementation.loc[0, 'truths'])}；每组重复：{int(implementation.loc[0, 'repeats_per_truth_min'])}。
- 有限估计比例：{implementation.loc[0, 'finite_estimate_fraction']:.3f}。
- 被选最佳拟合成功比例：{implementation.loc[0, 'selected_fit_success_fraction']:.3f}。
- 12个参数的真值层级Spearman相关范围：{rho_min:.3f}至{rho_max:.3f}；逐参数点估计和95% bootstrap区间见`T5/parameter-summary.csv`。
- 真值组内恢复误差的最大绝对参数间相关为{abs(max_error_correlation):.3f}（{max_error_pair[0]}与{max_error_pair[1]}，方向为{max_error_correlation:.3f}）；完整矩阵用于透明呈现估计权衡，不设事后删除规则。

## 证据边界

本次T5工作是对既有模拟结果的领域通用化重新汇总，不改变数据生成、拟合函数或原始估计。它可以替代旧的项目自定等级作为正式报告基础；若逐真值图或混淆矩阵暴露局部问题，再针对该问题决定是否需要增加模拟，而不是预先重复全部模拟。

## 方法依据

- Wilson & Collins (2019), *eLife*, DOI: 10.7554/eLife.49547：模拟参数范围匹配真实拟合范围，检查真值—恢复值关系、偏差和参数间混淆。
- Morris, White & Crowther (2019), *Statistics in Medicine*, DOI: 10.1002/sim.8086：按ADEMP组织模拟研究，同时报告多项性能指标和Monte Carlo误差。
"""
    appendix = """# 统计附录

## T5统计单位

- 误差：模拟数据集层级，并按真值分层。
- 排序恢复：每组真值60次估计的中位数，再跨16组真值计算Spearman相关。
- 相关区间：在每个真值组内对60次估计有放回重抽，重复2000次。
- 偏差Monte Carlo标准误：误差标准差除以模拟次数平方根。
- 参数混淆：同时报告真值相关、恢复中位数相关、真值—恢复值交叉相关，以及在每组真值内中心化后的恢复误差相关。这样可区分真值库自身相关与估计过程中出现的参数权衡。

## 不作出的声明

- 不把任一固定相关值或固定标准化偏差值称为领域通用通过标准。
- 不用总体相关掩盖逐真值或局部参数范围中的恢复问题。
- T6尚未产生正式恢复结果。
"""
    catalog = """# Figure catalog

## Figure 1: T5 truth versus recovered

- 文件：`figures/figure-01-t5-truth-vs-recovered.png` / `.pdf`
- 问题：每个参数在经验真值范围内是否保持真值—恢复值对应关系。
- 依据：16组真值各自60次恢复的中位数。
- 限制：点数由16组真值决定；精确数值与区间见参数汇总表。

## Figure 2: T5 parameter confusion

- 文件：`figures/figure-02-t5-parameter-confusion.png` / `.pdf`
- 问题：参数依赖来自真值库自身，还是来自恢复误差之间的权衡。
- 依据：并列展示真值相关、恢复中位数相关和真值组内恢复误差相关。
- 限制：相关提示依赖关系，不单独证明机制来源；完整真值—恢复值交叉矩阵另存CSV。
"""
    write_text(report, output_dir / "analysis-report.md")
    write_text(appendix, output_dir / "stats-appendix.md")
    write_text(catalog, output_dir / "figure-catalog.md")


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    truths, recovered, manifest = load_t5()
    summary, per_truth, matrices, implementation = summarize_t5(truths, recovered)
    rules = shared_rules()
    evidence_map = pd.DataFrame([
        {"task": "T5", "existing_simulation": "available", "field_common_resummary": "completed", "formal_outcomes": "available", "next_action": "review continuous metrics and figures"},
        {"task": "T6", "existing_simulation": "pilot only", "field_common_resummary": "not applicable", "formal_outcomes": "not generated", "next_action": "run frozen v2 formal design"},
    ])
    write_csv(summary, output_dir / "T5" / "parameter-summary.csv")
    write_csv(per_truth, output_dir / "T5" / "per-truth-performance.csv")
    for name, matrix in matrices.items():
        write_csv(matrix.rename_axis("parameter").reset_index(), output_dir / "T5" / f"{name}.csv")
    write_csv(implementation, output_dir / "T5" / "implementation-summary.csv")
    write_csv(rules, output_dir / "field-common-rules.csv")
    write_csv(evidence_map, output_dir / "t5-t6-evidence-map.csv")
    make_figures(per_truth, matrices, output_dir)
    render_reports(summary, implementation, matrices, output_dir)
    assert len(manifest) == 960
    return {
        "output_dir": str(output_dir),
        "t5_datasets": int(implementation.loc[0, "datasets"]),
        "t5_parameters": len(summary),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "t6_formal_outcomes_generated": False,
    }


if __name__ == "__main__":
    print(run())
