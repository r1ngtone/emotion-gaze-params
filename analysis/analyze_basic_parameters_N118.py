#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-basic-parameters-valid-n118-20260729")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
DATA = ROOT / "data"
T5_FILE = ROOT / "data" / "20260727_t5_118_valid_rerun" / "t5_results.csv"
T6_FILE = ROOT / "data" / "20260722_t6_118_binocular_rerun" / "t6_results.csv"

FIG_DIR = BASE / "figures"
TAB_DIR = BASE / "tables"
REPORT = BASE / "2026-07-29--T5-valid-N118基础参数分析报告.md"

IMAGE_TYPES = ["positive", "negative", "neutral", "complex"]
EMOTIONS = ["positive", "negative", "neutral"]
T5_PARAMS = ["T_half", "H_bar_norm", "I_focus"]
T6_RENAME = {
    "Pm1_positive": "pm_positive",
    "Pm2_negative": "pm_negative",
    "Pm3_neutral": "pm_neutral",
    "Tmax1": "Tmax_positive",
    "Tmax2": "Tmax_negative",
    "Tmax3": "Tmax_neutral",
    "类别": "predicted_class_t6",
}
T6_PARAMS = [
    "pm_positive",
    "pm_negative",
    "pm_neutral",
    "Tmax_positive",
    "Tmax_negative",
    "Tmax_neutral",
    "dc",
]

COLORS = {
    "positive": "#0072B2",
    "negative": "#D55E00",
    "neutral": "#999999",
    "complex": "#009E73",
    "t5": "#4C78A8",
    "t6": "#E69F00",
    "grid": "#E5E7EB",
    "text": "#222222",
}


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t5 = pd.read_csv(T5_FILE, encoding="utf-8-sig")
    t6_raw = pd.read_csv(T6_FILE, encoding="utf-8-sig")
    t6 = t6_raw.rename(columns=T6_RENAME)

    t5["subject_id"] = t5["subject_id"].astype(str)
    t6["subject_id"] = t6["subject_id"].astype(str)

    required_t5 = ["subject_id", "gender", "predicted_class"] + [f"{img}_{param}" for img in IMAGE_TYPES for param in T5_PARAMS]
    required_t6 = ["subject_id", "gender", "predicted_class_t6"] + T6_PARAMS
    missing_t5 = [col for col in required_t5 if col not in t5.columns]
    missing_t6 = [col for col in required_t6 if col not in t6.columns]
    if missing_t5 or missing_t6:
        raise ValueError(f"Missing required columns: T5={missing_t5}; T6={missing_t6}")
    if len(t5) != 118 or len(t6) != 118:
        raise ValueError(f"Expected 118 T5 and 118 T6 rows, got T5={len(t5)}, T6={len(t6)}")

    merged = t5[["subject_id", "gender", "predicted_class"]].merge(
        t6[["subject_id", "predicted_class_t6"] + T6_PARAMS],
        on="subject_id",
        how="inner",
    )
    if len(merged) != 118:
        raise ValueError(f"Expected 118 matched subjects, got {len(merged)}")

    numeric_cols = [f"{img}_{param}" for img in IMAGE_TYPES for param in T5_PARAMS] + T6_PARAMS
    numeric = pd.concat([t5[[col for col in numeric_cols if col in t5.columns]], t6[[col for col in numeric_cols if col in t6.columns]]], axis=1)
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad_cols = numeric.columns[numeric.isna().any() | ~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
        raise ValueError(f"Non-finite or missing numeric parameter values found in: {bad_cols}")
    return t5, t6, merged


def ci95_mean(values: pd.Series) -> tuple[float, float]:
    vals = values.dropna().astype(float)
    if len(vals) <= 1:
        return float("nan"), float("nan")
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    return float(vals.mean() - 1.96 * se), float(vals.mean() + 1.96 * se)


def describe_series(values: pd.Series) -> dict[str, float | int]:
    vals = values.dropna().astype(float)
    ci_low, ci_high = ci95_mean(vals)
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median": float(vals.median()),
        "iqr_low": float(vals.quantile(0.25)),
        "iqr_high": float(vals.quantile(0.75)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def build_t5_long(t5: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in t5.iterrows():
        for image_type in IMAGE_TYPES:
            for param in T5_PARAMS:
                rows.append(
                    {
                        "subject_id": row["subject_id"],
                        "gender": row["gender"],
                        "predicted_class": row["predicted_class"],
                        "task": "T5",
                        "condition": image_type,
                        "parameter": param,
                        "value": float(row[f"{image_type}_{param}"]),
                    }
                )
    return pd.DataFrame(rows)


def build_t6_long(t6: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in t6.iterrows():
        for emotion in EMOTIONS:
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "gender": row["gender"],
                    "predicted_class": row["predicted_class_t6"],
                    "task": "T6",
                    "condition": emotion,
                    "parameter": "pm",
                    "value": float(row[f"pm_{emotion}"]),
                }
            )
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "gender": row["gender"],
                    "predicted_class": row["predicted_class_t6"],
                    "task": "T6",
                    "condition": emotion,
                    "parameter": "Tmax",
                    "value": float(row[f"Tmax_{emotion}"]),
                }
            )
        rows.append(
            {
                "subject_id": row["subject_id"],
                "gender": row["gender"],
                "predicted_class": row["predicted_class_t6"],
                "task": "T6",
                "condition": "global",
                "parameter": "dc",
                "value": float(row["dc"]),
            }
        )
    return pd.DataFrame(rows)


def write_summary_tables(t5: pd.DataFrame, t6: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t5_long = build_t5_long(t5)
    t6_long = build_t6_long(t6)

    t5_summary = (
        t5_long.groupby(["task", "parameter", "condition"])["value"]
        .apply(lambda s: pd.Series(describe_series(s)))
        .unstack()
        .reset_index()
        .sort_values(["parameter", "condition"])
    )
    t6_summary = (
        t6_long.groupby(["task", "parameter", "condition"])["value"]
        .apply(lambda s: pd.Series(describe_series(s)))
        .unstack()
        .reset_index()
        .sort_values(["parameter", "condition"])
    )
    combined = pd.concat([t5_summary, t6_summary], ignore_index=True)

    t5_summary.to_csv(TAB_DIR / "t5_parameter_by_image_type_N118.csv", index=False, encoding="utf-8-sig")
    t6_summary.to_csv(TAB_DIR / "t6_parameter_by_emotion_N118.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(TAB_DIR / "parameter_descriptive_summary_N118.csv", index=False, encoding="utf-8-sig")
    return combined, t5_summary, t6_summary


def bh_fdr(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out.tolist()
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[valid])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[order] = np.clip(adjusted, 0, 1)
    return out.tolist()


def build_cross_task_correlations(t5: pd.DataFrame, t6: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t5_collapsed = pd.DataFrame({"subject_id": t5["subject_id"]})
    for param in T5_PARAMS:
        t5_collapsed[f"T5_mean_{param}"] = t5[[f"{img}_{param}" for img in IMAGE_TYPES]].mean(axis=1)
    corr_df = t5_collapsed.merge(t6[["subject_id"] + T6_PARAMS], on="subject_id", how="inner")

    rows: list[dict[str, object]] = []
    t5_cols = [f"T5_mean_{param}" for param in T5_PARAMS]
    for left in t5_cols:
        for right in T6_PARAMS:
            valid = corr_df[[left, right]].dropna()
            res = spearmanr(valid[left], valid[right])
            rho = float(getattr(res, "statistic", res.correlation))
            p = float(res.pvalue)
            rows.append(
                {
                    "t5_parameter": left,
                    "t6_parameter": right,
                    "n": int(len(valid)),
                    "spearman_rho": rho,
                    "p_value": p,
                }
            )
    corr_table = pd.DataFrame(rows)
    corr_table["p_fdr_bh"] = bh_fdr(corr_table["p_value"].tolist())
    corr_table["abs_spearman_rho"] = corr_table["spearman_rho"].abs()
    corr_table = corr_table.sort_values(["abs_spearman_rho", "p_fdr_bh"], ascending=[False, True])
    corr_table.to_csv(TAB_DIR / "t5_t6_cross_task_correlations_N118.csv", index=False, encoding="utf-8-sig")

    heat = corr_table.pivot(index="t5_parameter", columns="t6_parameter", values="spearman_rho").loc[t5_cols, T6_PARAMS]
    return corr_table, heat


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_t5_distributions(t5: pd.DataFrame) -> None:
    n_cols = len(T5_PARAMS)
    fig, axes = plt.subplots(1, n_cols, figsize=(max(7.2, n_cols * 2.8), 3.2), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, param in zip(axes, T5_PARAMS):
        data = [t5[f"{img}_{param}"].astype(float).to_numpy() for img in IMAGE_TYPES]
        parts = ax.violinplot(data, positions=np.arange(len(IMAGE_TYPES)), widths=0.75, showmeans=False, showextrema=False)
        for body, img in zip(parts["bodies"], IMAGE_TYPES):
            body.set_facecolor(COLORS[img])
            body.set_edgecolor("none")
            body.set_alpha(0.28)
        for i, vals in enumerate(data):
            rng = np.random.default_rng(100 + i)
            ax.scatter(
                np.full(len(vals), i) + rng.normal(0, 0.045, len(vals)),
                vals,
                s=7,
                color=COLORS[IMAGE_TYPES[i]],
                alpha=0.38,
                edgecolor="none",
            )
            ax.scatter(i, np.median(vals), s=28, marker="D", color="#222222", edgecolor="white", linewidth=0.4, zorder=4)
        ax.set_title(param)
        ax.set_xticks(np.arange(len(IMAGE_TYPES)))
        ax.set_xticklabels([x.capitalize() for x in IMAGE_TYPES], rotation=25, ha="right")
        ax.set_ylabel("Parameter value")
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    for ax in axes[len(T5_PARAMS) :]:
        ax.axis("off")
    fig.suptitle("Figure 1. T5 parameter distributions by image type", fontsize=11, fontweight="bold")
    save_figure(fig, "Figure_01_T5_parameter_distributions")


def plot_t6_distributions(t6: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), constrained_layout=True)
    specs = [
        ("pm", [f"pm_{emo}" for emo in EMOTIONS], "pm composition"),
        ("Tmax", [f"Tmax_{emo}" for emo in EMOTIONS], "Tmax (ms)"),
        ("dc", ["dc"], "dc"),
    ]
    for ax, (title, cols, ylabel) in zip(axes, specs):
        labels = [col.split("_", 1)[1].capitalize() if "_" in col else "Global" for col in cols]
        data = [t6[col].astype(float).to_numpy() for col in cols]
        parts = ax.violinplot(data, positions=np.arange(len(cols)), widths=0.72, showmeans=False, showextrema=False)
        for body, col in zip(parts["bodies"], cols):
            key = col.split("_", 1)[1] if "_" in col else "complex"
            body.set_facecolor(COLORS.get(key, COLORS["t6"]))
            body.set_edgecolor("none")
            body.set_alpha(0.30)
        for i, vals in enumerate(data):
            rng = np.random.default_rng(200 + i)
            key = cols[i].split("_", 1)[1] if "_" in cols[i] else "complex"
            ax.scatter(
                np.full(len(vals), i) + rng.normal(0, 0.045, len(vals)),
                vals,
                s=8,
                color=COLORS.get(key, COLORS["t6"]),
                alpha=0.42,
                edgecolor="none",
            )
            ax.scatter(i, np.median(vals), s=32, marker="D", color="#222222", edgecolor="white", linewidth=0.4, zorder=4)
        if title == "pm":
            ax.axhline(1 / 3, color="#666666", linestyle="--", linewidth=0.9)
            ax.text(0.02, 0.94, "Uniform share", transform=ax.transAxes, va="top", fontsize=7, color="#555555")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    fig.suptitle("Figure 2. T6 parameter distributions by emotion", fontsize=11, fontweight="bold")
    save_figure(fig, "Figure_02_T6_parameter_distributions")


def plot_heatmap(heat: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    im = ax.imshow(heat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = float(heat.iloc[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(val) > 0.45 else "#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Spearman rho")
    ax.set_title("Figure 3. Exploratory T5-T6 parameter correlation structure", fontsize=11, fontweight="bold")
    save_figure(fig, "Figure_03_parameter_correlation_heatmap")


def fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def fmt_p(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            if col in {"p_value", "p_fdr_bh"}:
                out[col] = out[col].map(lambda x: fmt_p(float(x)))
            else:
                out[col] = out[col].map(lambda x: fmt(float(x), 3))
    headers = list(out.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def make_report(
    t5: pd.DataFrame,
    t6: pd.DataFrame,
    combined: pd.DataFrame,
    t5_summary: pd.DataFrame,
    t6_summary: pd.DataFrame,
    corr_table: pd.DataFrame,
) -> None:
    gender_counts = t5["gender"].value_counts(dropna=False).rename_axis("gender").reset_index(name="n")
    t5_class = t5["predicted_class"].value_counts().rename_axis("class").reset_index(name="n")
    t6_class = t6["predicted_class_t6"].value_counts().rename_axis("class").reset_index(name="n")

    t5_means = t5_summary[["parameter", "condition", "mean", "sd", "median", "iqr_low", "iqr_high"]].copy()
    t6_means = t6_summary[["parameter", "condition", "mean", "sd", "median", "iqr_low", "iqr_high"]].copy()

    t5_highlights = []
    for param in T5_PARAMS:
        sub = t5_means[t5_means["parameter"] == param].sort_values("mean", ascending=False)
        top = sub.iloc[0]
        low = sub.iloc[-1]
        t5_highlights.append(f"`{param}` 均值最高的是 `{top['condition']}` ({fmt(top['mean'])})，最低的是 `{low['condition']}` ({fmt(low['mean'])})。")

    pm_sub = t6_means[t6_means["parameter"] == "pm"].sort_values("mean", ascending=False)
    tmax_sub = t6_means[t6_means["parameter"] == "Tmax"].sort_values("median", ascending=False)
    dc_row = t6_means[(t6_means["parameter"] == "dc") & (t6_means["condition"] == "global")].iloc[0]
    top_corr = corr_table.head(8)[["t5_parameter", "t6_parameter", "n", "spearman_rho", "p_value", "p_fdr_bh"]]

    lines = [
        "# N118 基础参数分析报告",
        "",
        "日期：2026-07-29",
        "",
        "## 1. 分析目标与数据",
        "",
        "本报告基于用户修改后的 118 人 T5/T6 参数表，对参数本身做第一步基础分析。分析范围包括参数分布、类别构成、T5 四类图片参数差异、T6 三类情绪参数差异，以及 T5-T6 参数之间的探索性 Spearman 相关结构。",
        "",
        "本报告不纳入量表关联、诊断解释、高低分组比较或模型恢复结论。T5参数来自 `20260727_t5_118_valid_rerun/t5_results.csv`，T6参数来自 `20260722_t6_118_binocular_rerun/t6_results.csv`，不使用历史 fit-record 参数值。当前 T5 数据只包含 `T_half/H_bar_norm/I_focus`，不再输出旧导出参数口径。",
        "",
        f"- T5 参数表：N={len(t5)}。",
        f"- T6 参数表：N={len(t6)}。",
        f"- T5/T6 按 `subject_id` 合并后：N={len(t5[['subject_id']].merge(t6[['subject_id']], on='subject_id'))}。",
        "",
        "性别结构仅作为样本描述：",
        "",
        markdown_table(gender_counts),
        "",
        "## 2. 主要输出文件",
        "",
        "- `tables/parameter_descriptive_summary_N118.csv`：T5/T6 参数描述统计总表。",
        "- `tables/t5_parameter_by_image_type_N118.csv`：T5 四类图片下各参数描述统计。",
        "- `tables/t6_parameter_by_emotion_N118.csv`：T6 三类情绪及全局参数描述统计。",
        "- `tables/t5_t6_cross_task_correlations_N118.csv`：T5 折叠参数与 T6 参数的探索性 Spearman 相关。",
        "- `figures/Figure_01_T5_parameter_distributions.*`：T5 参数分布图。",
        "- `figures/Figure_02_T6_parameter_distributions.*`：T6 参数分布图。",
        "- `figures/Figure_03_parameter_correlation_heatmap.*`：T5-T6 参数相关热图。",
        "",
        "## 3. T5 参数分布结论",
        "",
        "![Figure 1](figures/Figure_01_T5_parameter_distributions.png)",
        "",
        "图 1 回答的问题是：T5 单图任务中，四类图片条件下的三类当前参数是否呈现可见的分布差异和个体差异。每个小图展示一个参数；散点是被试，菱形为中位数，小提琴形状表示分布密度。",
        "",
        "从代码输出的描述统计看：",
        "",
        *[f"- {item}" for item in t5_highlights],
        "",
        "这些结果说明，T5 参数在四类图片之间存在不同程度的分布差异，同时每个参数都有明显的被试间差异。当前证据支持把 T5 参数作为个体化注意加工特征的描述变量，但这里尚未检验这些差异与量表或临床状态的关系。",
        "",
        "T5 描述统计摘录：",
        "",
        markdown_table(t5_means, max_rows=12),
        "",
        "## 4. T6 参数分布结论",
        "",
        "![Figure 2](figures/Figure_02_T6_parameter_distributions.png)",
        "",
        "图 2 回答的问题是：T6 多图任务的 `pm`、`Tmax` 和 `dc` 参数在 118 人样本中如何分布。`pm` 面板中的虚线是三类情绪均分时的 1/3 参考线。",
        "",
        f"- `pm` 均值最高的是 `{pm_sub.iloc[0]['condition']}` ({fmt(pm_sub.iloc[0]['mean'])})，最低的是 `{pm_sub.iloc[-1]['condition']}` ({fmt(pm_sub.iloc[-1]['mean'])})。",
        f"- `Tmax` 中位数最高的是 `{tmax_sub.iloc[0]['condition']}` ({fmt(tmax_sub.iloc[0]['median'])} ms)，最低的是 `{tmax_sub.iloc[-1]['condition']}` ({fmt(tmax_sub.iloc[-1]['median'])} ms)。",
        f"- `dc` 为全局参数，均值为 {fmt(dc_row['mean'])}，中位数为 {fmt(dc_row['median'])}，IQR 为 [{fmt(dc_row['iqr_low'])}, {fmt(dc_row['iqr_high'])}]。",
        "",
        "这些结果说明，T6 当前参数表可以同时给出资源占比、时间峰值和整体空间尺度三个层面的被试级刻画。`pm` 的三类情绪分布接近但不完全相同，`Tmax` 则更直接体现三类情绪达到模型峰值时间的差异；`dc` 反映整体空间尺度，不对应单一情绪类别。",
        "",
        "T6 描述统计摘录：",
        "",
        markdown_table(t6_means),
        "",
        "## 5. 类别构成",
        "",
        "T5 预测类别：",
        "",
        markdown_table(t5_class),
        "",
        "T6 预测类别：",
        "",
        markdown_table(t6_class),
        "",
        "类别构成可用于检查样本在模型分类标签上的总体分布。这里不把类别标签解释为诊断分组，也不据此做临床判断。",
        "",
        "## 6. T5-T6 参数关系",
        "",
        "![Figure 3](figures/Figure_03_parameter_correlation_heatmap.png)",
        "",
        "图 3 回答的问题是：把 T5 四类图片参数先按被试取均值后，它们与 T6 当前参数之间是否存在探索性的单调关系。颜色和数字均表示 Spearman rho。",
        "",
        "绝对相关最高的若干组合如下：",
        "",
        markdown_table(top_corr),
        "",
        "这些相关只说明两个任务的参数在被试层面可能共享部分个体差异结构。由于本报告没有纳入量表、诊断、实验操纵或纵向数据，因此不能把这些相关解释为因果关系、诊断标志或机制证明。",
        "",
        "## 7. 结论与边界",
        "",
        "当前 118 人参数表已经足以支持第一步参数基础分析：T5/T6 均为 118 行，关键参数无缺失或非有限值，按被试合并后仍为 118 人。T5 参数展示了四类图片下的分布差异与个体差异；T6 参数展示了 `pm`、`Tmax`、`dc` 三个层面的快速多图任务刻画；T5-T6 Spearman 热图提供了后续选择重点参数关系的探索依据。",
        "",
        "后续如果进入第二步，可以在这个基础上增加量表关联、组别比较或与 direct model-free 指标的对照，但需要单独处理多重比较、效应量和非诊断边界。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_plotting()
    ensure_dirs()
    t5, t6, _ = load_data()
    combined, t5_summary, t6_summary = write_summary_tables(t5, t6)
    corr_table, heat = build_cross_task_correlations(t5, t6)
    plot_t5_distributions(t5)
    plot_t6_distributions(t6)
    plot_heatmap(heat)
    make_report(t5, t6, combined, t5_summary, t6_summary, corr_table)
    print(f"Generated analysis bundle: {BASE}")


if __name__ == "__main__":
    main()
