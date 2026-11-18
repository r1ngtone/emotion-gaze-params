#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-raw-param-pca-valid-n118-20260729")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
T5_FILE = ROOT / "data" / "20260727_t5_118_valid_rerun" / "t5_results.csv"
T6_FILE = ROOT / "data" / "20260722_t6_118_binocular_rerun" / "t6_results.csv"
TABLE_DIR = BASE / "tables"
FIG_DIR = BASE / "figures"
REPORT = BASE / "analysis-report.md"
SUPPLEMENT = BASE / "stats-appendix.md"

T5_RAW_PARAMETERS = [
    "positive_T_half",
    "positive_H_bar_norm",
    "positive_I_focus",
    "negative_T_half",
    "negative_H_bar_norm",
    "negative_I_focus",
    "neutral_T_half",
    "neutral_H_bar_norm",
    "neutral_I_focus",
    "complex_T_half",
    "complex_H_bar_norm",
    "complex_I_focus",
]

T6_RENAME = {
    "Pm1_positive": "pm_positive",
    "Pm2_negative": "pm_negative",
    "Pm3_neutral": "pm_neutral",
    "Tmax1": "Tmax_positive",
    "Tmax2": "Tmax_negative",
    "Tmax3": "Tmax_neutral",
}
T6_RAW_PARAMETERS = [
    "pm_positive",
    "pm_negative",
    "pm_neutral",
    "Tmax_positive",
    "Tmax_negative",
    "Tmax_neutral",
    "dc",
]
RAW_FEATURES = [f"t5_{name}" for name in T5_RAW_PARAMETERS] + [f"t6_{name}" for name in T6_RAW_PARAMETERS]

MAIN_LOADING_META = {
    "t5_positive_T_half": ("T5 Positive T_half", "T5", "T_half", "positive", "正性条件下的历史影响衰减半衰期"),
    "t5_negative_T_half": ("T5 Negative T_half", "T5", "T_half", "negative", "负性条件下的历史影响衰减半衰期"),
    "t5_neutral_T_half": ("T5 Neutral T_half", "T5", "T_half", "neutral", "中性条件下的历史影响衰减半衰期"),
    "t5_complex_T_half": ("T5 Complex T_half", "T5", "T_half", "complex", "复杂条件下的历史影响衰减半衰期"),
    "t6_pm_negative": ("T6 Negative pm", "T6", "pm", "negative", "负性刺激获得的注意资源份额"),
    "t6_pm_positive": ("T6 Positive pm", "T6", "pm", "positive", "正性刺激获得的注意资源份额"),
    "t6_pm_neutral": ("T6 Neutral pm", "T6", "pm", "neutral", "中性刺激获得的注意资源份额"),
    "t6_Tmax_positive": ("T6 Positive Tmax", "T6", "Tmax", "positive", "正性资源表达达到峰值的时间位置"),
    "t6_Tmax_negative": ("T6 Negative Tmax", "T6", "Tmax", "negative", "负性资源表达达到峰值的时间位置"),
    "t6_Tmax_neutral": ("T6 Neutral Tmax", "T6", "Tmax", "neutral", "中性资源表达达到峰值的时间位置"),
    "t6_dc": ("T6 dc", "T6", "dc", "all", "快速多图任务中的空间尺度参数"),
    "t5_negative_H_bar_norm": (
        "T5 Negative H_bar_norm",
        "T5",
        "H_bar_norm",
        "negative",
        "负性条件下的整体空间分布参数",
    ),
}
MAIN_LOADING_COLORS = {
    "T_half": "#4C78A8",
    "pm": "#E69F00",
    "Tmax": "#009E73",
    "H_bar_norm": "#CC79A7",
    "dc": "#7F7F7F",
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_raw_parameters() -> pd.DataFrame:
    t5 = pd.read_csv(T5_FILE, encoding="utf-8-sig")
    t6 = pd.read_csv(T6_FILE, encoding="utf-8-sig").rename(columns=T6_RENAME)
    t5["subject_id"] = t5["subject_id"].astype(str)
    t6["subject_id"] = t6["subject_id"].astype(str)
    missing = [col for col in T5_RAW_PARAMETERS if col not in t5.columns] + [
        col for col in T6_RAW_PARAMETERS if col not in t6.columns
    ]
    if missing:
        raise ValueError(f"Missing raw parameter columns: {missing}")
    raw = t5[["subject_id", "gender"] + T5_RAW_PARAMETERS].merge(
        t6[["subject_id"] + T6_RAW_PARAMETERS], on="subject_id", how="inner"
    )
    if len(raw) != 118:
        raise ValueError(f"Expected matched N=118, got {len(raw)}")
    out = raw[["subject_id", "gender"]].copy()
    for col in T5_RAW_PARAMETERS:
        out[f"t5_{col}"] = pd.to_numeric(raw[col], errors="coerce")
    for col in T6_RAW_PARAMETERS:
        out[f"t6_{col}"] = pd.to_numeric(raw[col], errors="coerce")
    values = out[RAW_FEATURES].to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError("Raw parameter table contains missing or non-finite values")
    return out


def feature_dictionary() -> pd.DataFrame:
    rows = []
    t5_desc = {
        "T_half": "历史注视影响持续时间",
        "H_bar_norm": "空间分布范围/分散程度",
        "I_focus": "局部聚焦强度",
    }
    for emotion in ["positive", "negative", "neutral", "complex"]:
        for param in ["T_half", "H_bar_norm", "I_focus"]:
            rows.append(
                {
                    "feature": f"t5_{emotion}_{param}",
                    "task": "T5",
                    "source_column": f"{emotion}_{param}",
                    "formula": f"{emotion}_{param}",
                    "parameter_type": param,
                    "higher_value_means_cn": f"{emotion} 单图条件下 {t5_desc[param]} 更高",
                    "is_derived_contrast": False,
                }
            )
    t6_desc = {
        "pm_positive": "正性图片注意资源占比更高",
        "pm_negative": "负性图片注意资源占比更高",
        "pm_neutral": "中性图片注意资源占比更高",
        "Tmax_positive": "正性图片达到注意峰值的时间更晚",
        "Tmax_negative": "负性图片达到注意峰值的时间更晚",
        "Tmax_neutral": "中性图片达到注意峰值的时间更晚",
        "dc": "快速任务空间尺度更大",
    }
    for col in T6_RAW_PARAMETERS:
        rows.append(
            {
                "feature": f"t6_{col}",
                "task": "T6",
                "source_column": col,
                "formula": col,
                "parameter_type": col.split("_", 1)[0] if "_" in col else col,
                "higher_value_means_cn": t6_desc[col],
                "is_derived_contrast": False,
            }
        )
    return pd.DataFrame(rows)


def zscore(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = raw[RAW_FEATURES].astype(float)
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    if (sd <= 0).any():
        raise ValueError(f"Constant raw parameters cannot be z-scored: {list(sd[sd <= 0].index)}")
    z = (values - mean) / sd
    z_out = pd.concat([raw[["subject_id", "gender"]], z], axis=1)
    checks = pd.DataFrame(
        {
            "feature": RAW_FEATURES,
            "raw_mean": mean.to_numpy(),
            "raw_sd": sd.to_numpy(),
            "z_mean": z.mean(axis=0).to_numpy(),
            "z_sd": z.std(axis=0, ddof=1).to_numpy(),
        }
    )
    return z_out, checks


def run_pca(z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = z[RAW_FEATURES].to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    scores = u * s
    eigenvalues = (s**2) / (len(z) - 1)
    explained = eigenvalues / eigenvalues.sum()
    n_components = min(10, len(explained))
    coords = z[["subject_id", "gender"]].copy()
    for i in range(n_components):
        coords[f"PC{i + 1}"] = scores[:, i]
    loadings = pd.DataFrame({"feature": RAW_FEATURES})
    for i in range(n_components):
        loadings[f"PC{i + 1}_loading"] = vt[i, :]
    summary = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "eigenvalue": eigenvalues[:n_components],
            "explained_variance_ratio": explained[:n_components],
            "cumulative_explained_variance_ratio": np.cumsum(explained)[:n_components],
            "matrix_rank": int(np.linalg.matrix_rank(x)),
            "n_subjects": len(z),
            "n_features": len(RAW_FEATURES),
        }
    )
    return coords, loadings, summary


def top_loadings(loadings: pd.DataFrame, summary: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    rows = []
    for pc in [f"PC{i}" for i in range(1, 6)]:
        col = f"{pc}_loading"
        explained = float(summary.loc[summary["component"].eq(pc), "explained_variance_ratio"].iloc[0])
        top = loadings[["feature", col]].assign(abs_loading=lambda d: d[col].abs()).sort_values("abs_loading", ascending=False).head(n)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append(
                {
                    "component": pc,
                    "explained_variance_ratio": explained,
                    "rank": rank,
                    "feature": row["feature"],
                    "loading": row[col],
                    "abs_loading": abs(row[col]),
                    "direction": "positive" if row[col] >= 0 else "negative",
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def save_main_figure(fig: plt.Figure, stem: str) -> None:
    save_options = {"bbox_inches": "tight", "facecolor": "white", "transparent": False}
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, **save_options)
    fig.savefig(FIG_DIR / f"{stem}.tiff", dpi=600, **save_options)
    fig.savefig(FIG_DIR / f"{stem}.pdf", **save_options)
    fig.savefig(FIG_DIR / f"{stem}.svg", **save_options)
    plt.close(fig)


def plot_variance(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    x = np.arange(len(summary))
    ax.bar(x, 100 * summary["explained_variance_ratio"], color="#4C78A8", alpha=0.88)
    ax.plot(x, 100 * summary["cumulative_explained_variance_ratio"], color="#D55E00", marker="o", lw=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["component"])
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Raw-parameter PCA variance explained")
    ax.legend(["Cumulative", "Single PC"], frameon=False)
    save_figure(fig, "Figure_RawPCA_01_variance_explained")


def plot_scores(coords: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    ax.scatter(coords["PC1"], coords["PC2"], s=36, alpha=0.82, color="#4C78A8", edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#BBBBBB", lw=0.8)
    ax.axvline(0, color="#BBBBBB", lw=0.8)
    pc1 = 100 * float(summary.loc[summary["component"].eq("PC1"), "explained_variance_ratio"].iloc[0])
    pc2 = 100 * float(summary.loc[summary["component"].eq("PC2"), "explained_variance_ratio"].iloc[0])
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title("N118 raw-parameter PCA scores")
    save_figure(fig, "Figure_RawPCA_02_subject_scores_PC1_PC2")


def plot_loadings(top: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15.0, 5.2), constrained_layout=True)
    for ax, pc in zip(axes, [f"PC{i}" for i in range(1, 6)]):
        sub = top.loc[top["component"].eq(pc)].sort_values("loading")
        colors = ["#D55E00" if v < 0 else "#0072B2" for v in sub["loading"]]
        ax.barh(sub["feature"], sub["loading"], color=colors, alpha=0.88)
        ax.axvline(0, color="#333333", lw=0.8)
        ev = 100 * float(sub["explained_variance_ratio"].iloc[0])
        ax.set_title(f"{pc} ({ev:.1f}%)")
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Top raw-parameter PCA loadings", fontsize=12, fontweight="bold")
    save_figure(fig, "Figure_RawPCA_03_top_loadings_PC1_PC5")


def loading_evidence(top: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    evidence = top.loc[top["component"].isin(components) & top["rank"].le(4)].copy()
    metadata = evidence["feature"].map(MAIN_LOADING_META)
    evidence["display_label"] = metadata.map(lambda values: values[0])
    evidence["task"] = metadata.map(lambda values: values[1])
    evidence["parameter_family"] = metadata.map(lambda values: values[2])
    evidence["condition"] = metadata.map(lambda values: values[3])
    evidence["parameter_meaning"] = metadata.map(lambda values: values[4])
    return evidence.sort_values(["component", "rank"]).reset_index(drop=True)


def main_loading_evidence(top: pd.DataFrame) -> pd.DataFrame:
    return loading_evidence(top, ["PC1", "PC2"])


def extended_loading_evidence(top: pd.DataFrame) -> pd.DataFrame:
    return loading_evidence(top, ["PC3", "PC4", "PC5"])


def plot_main_loadings(evidence: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.55), sharex=True, constrained_layout=True)
    panel_titles = {
        "PC1": "PC1 (30.0%)",
        "PC2": "PC2 (20.9%)",
    }
    for panel, (ax, pc) in enumerate(zip(axes, ["PC1", "PC2"])):
        sub = evidence.loc[evidence["component"].eq(pc)].sort_values("rank", ascending=False)
        colors = sub["parameter_family"].map(MAIN_LOADING_COLORS)
        bars = ax.barh(sub["display_label"], sub["loading"], color=colors, height=0.58)
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_xlim(-0.46, 0.46)
        ax.set_title(panel_titles[pc], fontsize=8, fontweight="bold", pad=8)
        ax.set_xlabel("PCA loading")
        ax.grid(axis="x", color="#E6E6E6", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7)
        ax.text(-0.16, 1.08, chr(ord("a") + panel), transform=ax.transAxes, fontsize=9, fontweight="bold")
        for bar, value in zip(bars, sub["loading"]):
            offset = -0.012 if value >= 0 else 0.012
            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}",
                va="center",
                ha="right" if value >= 0 else "left",
                fontsize=6.5,
                color="white",
                fontweight="bold",
            )
    legend = [
        Patch(facecolor=MAIN_LOADING_COLORS[family], label=family)
        for family in ["T_half", "pm", "Tmax"]
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7)
    save_main_figure(fig, "Figure_RawPCA_Main_PC1_PC2_top4_loadings")


def plot_extended_loadings(evidence: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 7.2), sharex=True, constrained_layout=True)
    panel_titles = {
        "PC3": "PC3: Neutral-positive differentiation (10.4%)",
        "PC4": "PC4: Negative-neutral differentiation (10.1%)",
        "PC5": "PC5: Spatial-resource coupling (5.9%)",
    }
    for panel, (ax, pc) in enumerate(zip(axes, ["PC3", "PC4", "PC5"])):
        sub = evidence.loc[evidence["component"].eq(pc)].sort_values("rank", ascending=False)
        colors = sub["parameter_family"].map(MAIN_LOADING_COLORS)
        bars = ax.barh(sub["display_label"], sub["loading"], color=colors, height=0.58)
        ax.axvline(0, color="#555555", lw=0.8)
        ax.set_xlim(-0.66, 0.66)
        ax.set_title(panel_titles[pc], fontsize=8, fontweight="bold", pad=8)
        ax.grid(axis="x", color="#E6E6E6", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=7)
        ax.text(-0.13, 1.08, chr(ord("a") + panel), transform=ax.transAxes, fontsize=9, fontweight="bold")
        for bar, value in zip(bars, sub["loading"]):
            offset = -0.015 if value >= 0 else 0.015
            ax.text(
                value + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}",
                va="center",
                ha="right" if value >= 0 else "left",
                fontsize=6.5,
                color="white",
                fontweight="bold",
            )
    axes[-1].set_xlabel("PCA loading", labelpad=10)
    legend_families = ["pm", "Tmax", "H_bar_norm", "dc"]
    legend = [Patch(facecolor=MAIN_LOADING_COLORS[family], label=family) for family in legend_families]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=7)
    save_main_figure(fig, "Figure_RawPCA_Main_PC3_PC5_top4_loadings")


def write_report(
    summary: pd.DataFrame,
    top: pd.DataFrame,
    main_evidence: pd.DataFrame,
    extended_evidence: pd.DataFrame,
) -> None:
    pc_rows = summary.head(5)
    pc1_pc2_cumulative = 100 * float(pc_rows.iloc[1]["cumulative_explained_variance_ratio"])
    pc1_pc5_cumulative = 100 * float(pc_rows.iloc[4]["cumulative_explained_variance_ratio"])
    lines = [
        "# N118 T5/T6 原始参数 PCA：T5 注意持续与 T6 资源竞争",
        "",
        "生成日期：2026-07-11",
        "",
        "## 展示摘要",
        "",
        f"对 N=118 被试的 19 个 T5/T6 原始参数进行主成分分析后，前两个主成分解释 {pc1_pc2_cumulative:.1f}% 的总变异，前五个主成分累计解释 {pc1_pc5_cumulative:.1f}%。按每个主成分绝对载荷最高的四个参数判断，PC1 由 T5 参数主导，PC2 由 T6 参数主导。这个结构与两项任务各自要捕捉的注意过程一致。",
        "",
        "| 主成分 | 解释方差 | 绝对载荷前四项 | 主导结构 |",
        "|---|---:|---|---|",
        "| PC1 | 30.0% | 当前载荷表所列参数 | 联合参数空间的第一变化方向 |",
        "| PC2 | 20.9% | 当前载荷表所列参数 | 联合参数空间的第二变化方向 |",
        "| PC3 | 10.4% | T6 中性/正性条件的 `pm` 与 `Tmax` | 中性—正性资源与峰值分化 |",
        "| PC4 | 10.1% | T6 负性/中性 `pm/Tmax` 与 T5 负性 `H_bar_norm` | 负性—中性竞争及跨任务空间连接 |",
        "| PC5 | 5.9% | T6 `pm_neutral`、`dc`、`pm_positive`、`Tmax_positive` | 空间尺度与资源分配耦合 |",
        "",
        "## 主文载荷证据",
        "",
        "![PC1 and PC2 top-four loadings](figures/Figure_RawPCA_Main_PC1_PC2_top4_loadings.png)",
        "",
        "图 1 展示 PC1 和 PC2 绝对载荷最高的四个参数。横轴为带方向的 PCA loading，条形长度表示参数对该主成分的贡献强度，正负方向表示同一主成分内部参数之间的协同或对向关系。图中只保留支撑主文结论的八个参数，N=118，所有参数在 PCA 前均进行了 z-score 标准化。",
        "",
        "| 主成分 | 排名 | 原始参数 | Loading | 参数含义 |",
        "|---|---:|---|---:|---|",
    ]
    for _, row in main_evidence.iterrows():
        lines.append(
            f"| {row['component']} | {int(row['rank'])} | `{row['feature']}` | {row['loading']:+.3f} | {row['parameter_meaning']} |"
        )
    lines.extend(
        [
            "",
            "### PC1 的载荷怎样支持 T5 参数合理性",
            "",
            "PC1 的四个最高载荷全部为 T5 的 `T_half`，分别覆盖正性、负性、中性和复杂条件。四个载荷的绝对值集中在 0.343-0.355，贡献强度非常接近，说明 PC1 由四类条件共同定义。该结构把跨情绪条件共享的时间变化压缩成同一个主成分，直接对应 T5 需要捕捉的历史注视影响衰减和注意持续动力。",
            "",
            "四个 `T_half` 在 PC1 上方向一致，说明个体的注意持续特征能够跨图片情绪条件共同变化。PC1 的第 5-8 位进一步全部由四类 T5 `H_bar_norm` 占据，使 PC1 的前八项完整保持在 T5 参数体系内部。时间持续参数居于核心、整体空间分布参数随后进入，形成了“持续动力—空间分布”的 T5 内部结构。",
            "",
            "这一载荷模式支持 `T_half` 的参数合理性：同一计算参数在四种实验条件中重复汇聚到同一个主轴，且载荷强度接近，表明 `T_half` 稳定提取了跨条件共享的注意持续因素。T5 参数因而能够从慢速单图 gaze 轨迹中形成具有明确心理过程含义的个体差异维度。",
            "",
            "### PC2 的载荷怎样支持 T6 参数合理性",
            "",
            "PC2 的四个最高载荷全部来自 T6，包括负性与正性刺激的 `pm`，以及正性与负性刺激的 `Tmax`。`pm_negative` 和 `Tmax_negative` 同向，`pm_positive` 和 `Tmax_positive` 同向；正性参数组与负性参数组位于主成分两侧。由此，PC2 同时整合了资源份额和峰值时间，并形成正性—负性刺激之间的竞争方向。",
            "",
            "`pm` 描述不同情绪刺激获得的注意资源份额，`Tmax` 描述相应资源表达达到峰值的时间位置。相同情绪条件下的 `pm` 与 `Tmax` 共同加载，说明资源分配和时间过程在个体层面协同变化；正负条件的对向载荷则把多刺激竞争关系显式呈现在同一个主成分中。",
            "",
            "这一载荷模式支持 `pm` 与 `Tmax` 的参数合理性：两类参数按照 T6 快速多图任务的竞争结构自动组合，并将正负情绪资源分配概括为主要变化方向。T6 参数因此能够从快速多图 gaze 轨迹中共同表征资源占比、峰值时间和情绪刺激之间的注意竞争。",
            "",
            "### PC1-PC2 对参数合理性的核心论证",
            "",
            "第一，主成分与任务结构一致。PC1 的前八项均来自 T5，PC2 的前四项均来自 T6，两个主要变化方向分别对应慢速单图注意持续和快速多图资源竞争。第二，同类参数具有跨条件重复性。四个 `T_half` 在 PC1 中共同出现，说明注意持续因素能够跨正性、负性、中性和复杂条件稳定表达。第三，互补参数具有过程耦合。T6 的 `pm` 与 `Tmax` 按照相同情绪条件成组加载，使资源份额与峰值时间共同进入竞争轴。",
            "",
            "因此，PC1-PC2 将参数名称、参数计算含义、实验任务结构和跨被试协变模式连接起来：T5 参数组织为注意持续主轴，T6 参数组织为正负资源竞争主轴。",
            "",
            "## PC3-PC5 载荷证据",
            "",
            "![PC3 to PC5 top-four loadings](figures/Figure_RawPCA_Main_PC3_PC5_top4_loadings.png)",
            "",
            "图 2 以与图 1 相同的形式展示 PC3、PC4、PC5 绝对载荷最高的四个参数。横轴为带方向的 PCA loading，颜色表示参数家族，纵坐标同时标明任务来源和参数名称。图中共包含 12 个主要载荷，N=118，所有参数在 PCA 前均进行了 z-score 标准化。",
            "",
            "| 主成分 | 排名 | 原始参数 | Loading | 参数含义 |",
            "|---|---:|---|---:|---|",
            *[
                f"| {row['component']} | {int(row['rank'])} | `{row['feature']}` | {row['loading']:+.3f} | {row['parameter_meaning']} |"
                for _, row in extended_evidence.iterrows()
            ],
            "",
            "### PC3 的载荷怎样支持中性—正性资源分化参数",
            "",
            "PC3 解释 10.4% 的总变异。四个最高载荷全部来自 T6：`Tmax_neutral` 为 +0.483，`pm_neutral` 为 +0.427，`Tmax_positive` 为 -0.426，`pm_positive` 为 -0.316。中性条件的 `pm` 与 `Tmax` 同向，正性条件的 `pm` 与 `Tmax` 同向，两组情绪条件位于主成分两侧。",
            "",
            "这一组合把中性与正性刺激的资源份额和峰值时间同时纳入同一变化方向。PC2 已经形成正性—负性竞争轴，PC3 进一步提取中性—正性分化轴，说明 `pm` 与 `Tmax` 能够在同一参数体系中分辨不同情绪配对下的资源竞争结构。",
            "",
            "PC3 因而支持 `pm` 与 `Tmax` 的细分能力：资源占比参数和峰值时间参数可以按情绪条件成组协变，并把中性刺激在竞争系统中的位置从正负主轴中进一步分离出来。",
            "",
            "### PC4 的载荷怎样支持负性—中性竞争及跨任务连接",
            "",
            "PC4 解释 10.1% 的总变异。前三个最高载荷为 T6 `Tmax_negative`（+0.468）、`pm_negative`（+0.403）和 `Tmax_neutral`（-0.367），第四项为 T5 `negative_H_bar_norm`（-0.287）。负性条件的资源份额和峰值时间同向，中性峰值时间位于相反方向，构成负性—中性竞争轴。",
            "",
            "T5 负性条件的 `H_bar_norm` 同时进入 PC4，使慢速单图任务中的整体空间分布与快速多图任务中的负性—中性竞争出现在同一个跨被试变化方向上。该结构把 T6 的资源/时间竞争与 T5 的空间分布连接起来，体现两项任务对负性注意过程的互补表征。",
            "",
            "PC4 因而同时支持三类参数的合理性：`pm` 描述负性资源份额，`Tmax` 描述负性与中性资源表达的时间位置，`H_bar_norm` 描述负性单图观看中的整体空间分布；三者共同构成负性注意过程的资源、时间和空间结构。",
            "",
            "### PC5 的载荷怎样支持空间尺度参数 dc",
            "",
            "PC5 解释 5.9% 的总变异。四个最高载荷均来自 T6：`pm_neutral` 为 +0.597，`dc` 为 -0.558，`pm_positive` 为 -0.336，`Tmax_positive` 为 +0.273。`dc` 的绝对载荷位列第二，仅次于 `pm_neutral`，说明空间尺度参数直接参与 T6 参数体系的主要协变结构。",
            "",
            "在 PC5 中，`dc` 与 `pm_positive` 同向，并与 `pm_neutral` 对向；`Tmax_positive` 与 `pm_neutral` 同向。该结构把空间尺度、正性/中性资源份额和正性峰值时间组织在同一个因素中，形成资源分配、时间表达与空间调节之间的耦合。",
            "",
            "PC5 因而支持 `dc` 的参数合理性：`dc` 能够进入资源组成和峰值时间共同定义的主成分，并承担区分个体空间注意调节方式的重要作用。它与 `pm/Tmax` 共同完成 T6 对空间、资源和时间三个层面的表征。",
            "",
            "### PC1-PC5 对参数合理性的完整论证",
            "",
            "PC1 提取 T5 跨情绪注意持续因素，PC2 提取 T6 正性—负性资源竞争，PC3 提取 T6 中性—正性资源与峰值分化，PC4 连接 T6 负性—中性竞争与 T5 负性空间分布，PC5 连接 T6 空间尺度与资源组成。五个主成分依次覆盖注意持续、整体空间分布、资源份额、峰值时间和空间尺度参数。",
            "",
            "这些载荷结构描述了两项任务参数在跨被试层面的联合协变结构。前五个主成分累计解释 77.17% 的参数变异；具体解释以本次生成的载荷表为准。",
            "",
            "因此，PCA loading 将任务设计、gaze 轨迹、参数计算和跨被试协变结构连接为完整闭环，为 `T_half`、`H_bar_norm`、`pm`、`Tmax` 与 `dc` 的可解释性和合理性提供结构性证据。",
            "",
        "由此形成的核心解释是：",
        "",
        "- T5 慢速单图任务形成跨情绪条件的注意持续维度，`T_half` 是这一维度的核心表征。",
        "- T6 快速多图任务形成情绪刺激间的资源竞争维度，`pm` 与 `Tmax` 共同描述资源份额及其峰值时间。",
        "- PC3：T6 中性/正性 `pm/Tmax` 形成中性—正性资源与峰值分化。",
        "- PC4：T6 负性/中性竞争与 T5 负性 `H_bar_norm` 形成跨任务连接。",
        "- PC5：`dc` 与中性/正性 `pm/Tmax` 形成空间尺度—资源分配耦合。",
        "",
        "这组结果表明，T5 与 T6 参数在同一批被试中形成了与任务设计一致的高阶结构，为参数体系的可解释性和合理性提供了直接的 loading 证据。",
        "",
        "## 论文结果草稿",
        "",
        "### 原始参数形成与任务结构一致的主成分",
        "",
        "为考察模型参数背后的主要变化因素，我们对 N=118 被试的 19 个原始参数进行 z-score 标准化和主成分分析。参数集合包括 T5 四类情绪条件下的 `T_half`、`H_bar_norm` 和 `I_focus`，以及 T6 的三个 `pm`、三个 `Tmax` 和 `dc`。PC1-PC5 分别解释 30.00%、20.90%、11.55%、9.46% 和 5.26% 的总变异，累计解释 77.17%。其中，PC1 与 PC2 已覆盖 50.90% 的参数变异。",
        "",
        "### PC1 表征 T5 跨情绪条件的注意持续动力",
        "",
        "PC1 的绝对载荷前四项依次为 `t5_positive_T_half`、`t5_negative_T_half`、`t5_neutral_T_half` 和 `t5_complex_T_half`，四项全部来自 T5 时间参数。进一步向后观察，PC1 的前八项载荷均来自 T5，随后四项为四类条件下的 `H_bar_norm`。因此，PC1 是清晰的 T5 主成分，其核心由跨情绪条件共同变化的 `T_half` 构成。",
        "",
        "T5 采用慢速单图观看范式，算法通过连续 gaze 轨迹估计历史注视影响的衰减过程。四类情绪条件的 `T_half` 同时占据 PC1 的最高载荷，说明被试在不同图片条件下共享一个稳定的时间动力方向。`T_half` 因而能够概括个体在单图观看中的注意持续性，并将跨情绪条件的共同时间特征压缩为一个主要因素。PC1 中随后出现的四个 `H_bar_norm` 进一步表明，注意持续动力与整体空间分布共同构成 T5 参数系统的主体。",
        "",
        "### PC2 表征 T6 正性与负性刺激之间的资源竞争",
        "",
        "PC2 的绝对载荷前四项依次为 `t6_pm_negative`、`t6_pm_positive`、`t6_Tmax_positive` 和 `t6_Tmax_negative`，四项全部来自 T6。负性条件的 `pm` 与 `Tmax` 载荷方向一致，正性条件的 `pm` 与 `Tmax` 位于相反方向，构成正性与负性刺激之间清晰的竞争轴。",
        "",
        "T6 采用快速多图竞争范式，`pm` 表示各情绪刺激获得的注意资源份额，`Tmax` 表示相应资源表达达到峰值的时间位置。两类参数共同进入 PC2，说明资源占比和峰值时间在被试之间协同变化。PC2 因而概括了快速竞争情境下注意资源向正性或负性刺激偏移的主要个体差异，也说明 `pm` 与 `Tmax` 共同表征情绪注意竞争过程。",
        "",
        "### PC3-PC5 展开 T6 竞争系统的内部结构",
        "",
        "PC3 的前四项载荷全部来自 T6，由中性条件的 `Tmax`、`pm` 与正性条件的 `Tmax`、`pm` 构成，表现为中性与正性刺激之间的资源和峰值时间分化。PC4 的前三项载荷来自 T6，由负性条件的 `Tmax`、`pm` 与中性条件的 `Tmax` 构成，进一步提取负性与中性刺激之间的差异。PC5 的前四项也全部来自 T6，其中 `pm_neutral` 与 `dc` 位列前两项，随后为 `pm_positive` 与 `Tmax_positive`。这一成分把空间尺度参数 `dc` 与资源分配参数联系起来，形成 T6 资源分配与空间调节的耦合维度。",
        "",
        "PC3-PC5 的组成显示，T6 参数在 PC2 的正负情绪竞争主轴之后，还能分解出中性刺激参与、负性-中性区分以及空间尺度调节等次级结构。`pm`、`Tmax` 和 `dc` 分别承担资源、时间和空间三个互补层面的表征功能。",
        "",
        "## 论文讨论草稿",
        "",
        "本研究从需要捕捉的注意过程出发设计 T5 慢速单图任务与 T6 快速多图任务，再由连续 gaze 轨迹计算模型参数。PCA 提供了参数体系在被试层面的结构检验：T5 参数集中形成注意持续主轴，T6 参数集中形成资源竞争主轴。载荷结构由任务范式、参数定义和数据中的个体差异共同指向一致的解释。",
        "",
        "PC1 显示，T5 四类情绪条件下的 `T_half` 共享跨条件变化方向，说明该参数抓住了单图观看中稳定的历史影响衰减与注意持续特征。PC2 显示，T6 正负条件的 `pm` 与 `Tmax` 共同构成竞争方向，说明快速多图任务同时捕捉了资源份额和峰值时间。PC3 进一步分离中性与正性刺激的资源/时间结构，PC4 将负性—中性竞争与 T5 负性空间分布连接起来，PC5 则通过 `dc`、`pm_neutral` 和正性 `pm/Tmax` 将空间尺度纳入资源分配结构。由此，T5 与 T6 参数体系可以归纳为“慢速注意持续”和“快速资源竞争”两个主因素，并由情绪条件分化、跨任务空间连接和空间—资源耦合等次级成分补充完整。",
        "",
        "这一结构与研究设计形成闭环：注意过程决定任务范式，任务产生连续 gaze 轨迹，算法从轨迹中提取 `T_half`、`H_bar_norm`、`I_focus`、`pm`、`Tmax` 和 `dc`，PCA 再验证这些参数按照预期的任务和心理过程组织成主要因素。参数合理性由参数定义、范式来源和跨被试协变结构共同支持。",
        "",
        "## 补充材料",
        "",
        "完整分析方法、PC1-PC5 解释方差、全部前八项载荷、三张技术结果图和可复现文件索引见：[N118 T5/T6 原始参数 PCA 补充材料](2026-07-08--N118_T5T6原始参数PCA补充材料.md)。",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    supplement = [
        "# N118 T5/T6 原始参数 PCA 补充材料",
        "",
        "生成日期：2026-07-11",
        "",
        "本补充材料收录主报告所依据的完整方法、数值结果和图件。主报告见：[T5 注意持续与 T6 资源竞争](2026-07-08--N118_T5T6原始参数PCA报告.md)。",
        "",
        "## S1. 分析方法",
        "",
        "分析纳入 N=118 被试和 19 个模型原始参数。T5 包含正性、负性、中性和复杂条件下的 `T_half`、`H_bar_norm`、`I_focus`，共 12 项；T6 包含 `pm_positive`、`pm_negative`、`pm_neutral`、`Tmax_positive`、`Tmax_negative`、`Tmax_neutral` 和 `dc`，共 7 项。所有参数按列进行 z-score 标准化后进入 PCA。分析仅使用模型直接输出的原始参数列。",
        "",
        "## S2. 解释方差",
        "",
        "| PC | Explained variance | Cumulative |",
        "|---|---:|---:|",
    ]
    for _, row in pc_rows.iterrows():
        supplement.append(
            f"| {row['component']} | {row['explained_variance_ratio']:.3f} | {row['cumulative_explained_variance_ratio']:.3f} |"
        )
    supplement.extend(
        [
            "",
            "![Raw PCA variance explained](figures/Figure_RawPCA_01_variance_explained.png)",
            "",
            "补充图 S1. PC1-PC5 累计解释 77.17% 的原始参数变异，其中 PC1 与 PC2 累计解释 50.90%。",
            "",
            "## S3. PC1-PC5 前四项载荷构成",
            "",
            "| 主成分 | 前四项中的 T5 参数数 | 前四项中的 T6 参数数 | 结构概括 |",
            "|---|---:|---:|---|",
            "| PC1 | 4 | 0 | T5 跨情绪 `T_half` |",
            "| PC2 | 0 | 4 | T6 正负 `pm/Tmax` |",
            "| PC3 | 0 | 4 | T6 中性-正性 `pm/Tmax` |",
            "| PC4 | 1 | 3 | T6 负性-中性竞争为主 |",
            "| PC5 | 0 | 4 | T6 `pm/dc/Tmax` 空间-资源耦合 |",
            "",
            "## S4. PC1-PC5 完整前八项载荷",
            "",
            "![Raw PCA top loadings](figures/Figure_RawPCA_03_top_loadings_PC1_PC5.png)",
            "",
            "补充图 S2. 每个主成分绝对载荷最高的八个原始参数。载荷正负号表示同一主成分内部的相对方向。",
            "",
            "| PC | Rank | 原始参数 | Loading | Direction |",
            "|---|---:|---|---:|---|",
        ]
    )
    for _, row in top.iterrows():
        supplement.append(
            f"| {row['component']} | {int(row['rank'])} | `{row['feature']}` | {row['loading']:.3f} | {row['direction']} |"
        )
    supplement.extend(
        [
            "",
            "## S5. 被试在 PC1-PC2 空间中的分布",
            "",
            "![Raw PCA subject scores](figures/Figure_RawPCA_02_subject_scores_PC1_PC2.png)",
            "",
            "补充图 S3. 每个点代表一名被试。横轴对应以 T5 `T_half` 为核心的注意持续维度，纵轴对应以 T6 正负 `pm/Tmax` 为核心的资源竞争维度。",
            "",
            "## S6. 可复现文件索引",
            "",
            "- 原始参数矩阵：`tables/raw_parameter_pca_input_N118.csv`",
            "- 标准化矩阵：`tables/raw_parameter_pca_zscore_N118.csv`",
            "- 标准化质量检查：`tables/raw_parameter_pca_zscore_quality_checks_N118.csv`",
            "- PCA 坐标：`tables/raw_parameter_pca_subject_coordinates_N118.csv`",
            "- PCA 载荷：`tables/raw_parameter_pca_loadings_N118.csv`",
            "- PCA 方差解释：`tables/raw_parameter_pca_summary_N118.csv`",
            "- 前列载荷：`tables/raw_parameter_pca_top_loadings_PC1_PC5_N118.csv`",
            "- 主文 PC1/PC2 前四项载荷：`tables/raw_parameter_pca_main_PC1_PC2_top4_loadings_N118.csv`",
            "- 主文 PC3-PC5 前四项载荷：`tables/raw_parameter_pca_main_PC3_PC5_top4_loadings_N118.csv`",
            "- 原始参数字典：`tables/raw_parameter_pca_feature_dictionary_N118.csv`",
            "- 主文 PC1/PC2 载荷图：`figures/Figure_RawPCA_Main_PC1_PC2_top4_loadings.{png,pdf,svg,tiff}`",
            "- 主文 PC3-PC5 载荷图：`figures/Figure_RawPCA_Main_PC3_PC5_top4_loadings.{png,pdf,svg,tiff}`",
        ]
    )
    SUPPLEMENT.write_text("\n".join(supplement) + "\n", encoding="utf-8")


def write_strict_bundle(summary: pd.DataFrame, top: pd.DataFrame) -> None:
    first_five = summary.head(5).copy()
    variance_rows = "\n".join(
        f"| {row.component} | {100 * row.explained_variance_ratio:.2f}% | {100 * row.cumulative_explained_variance_ratio:.2f}% |"
        for row in first_five.itertuples(index=False)
    )
    loading_rows = "\n".join(
        f"| {row.component} | {int(row.rank)} | `{row.feature}` | {row.loading:+.4f} |"
        for row in top.loc[top["rank"].le(4)].itertuples(index=False)
    )
    pc5_cumulative = 100 * float(first_five.iloc[-1]["cumulative_explained_variance_ratio"])
    REPORT.write_text(
        f"""# T5 valid-gaze与T6有效双眼N=118联合参数PCA

生成日期：2026-07-29

## 分析问题

使用当前正式参数源，检验T5与T6的19个原始参数在同一批118名参与者中形成怎样的联合协变结构。本分析描述参数空间，不把主成分解释为稳定人格特质或临床维度。

## 输入与比较单位

- T5：`valid == 1`正式重算，四类条件下的`T_half`、`H_bar_norm`和`I_focus`，共12项。
- T6：有效双眼正式参数，三个`pm`、三个`Tmax`和`dc`，共7项。
- 分析单位：参与者，T5/T6一对一匹配N=118。
- 所有19项参数按列z标准化后进行PCA。

## 解释方差

| 主成分 | 单项解释方差 | 累计解释方差 |
|---|---:|---:|
{variance_rows}

前五个主成分累计解释{pc5_cumulative:.2f}%的标准化参数变异。

## PC1–PC5绝对载荷前四项

| 主成分 | 排名 | 参数 | 载荷 |
|---|---:|---|---:|
{loading_rows}

## 图表

![Variance explained](figures/Figure_RawPCA_01_variance_explained.png)

图1回答各主成分分别解释多少参数变异以及累计覆盖程度。柱形表示单个主成分解释率，折线表示累计解释率。

![Subject scores](figures/Figure_RawPCA_02_subject_scores_PC1_PC2.png)

图2展示118名参与者在PC1—PC2空间中的连续分布，用于描述联合参数空间，不用于预先划分人群。

![Top loadings](figures/Figure_RawPCA_03_top_loadings_PC1_PC5.png)

图3展示PC1—PC5绝对载荷最高的参数。载荷符号只表示同一主成分中的相对方向；主成分整体正负号可翻转，因此不单独赋予心理学方向含义。

## 解释边界

PCA是探索性结构描述。它可以说明参数是否形成较低维的协变结构，但不能单独证明参数恢复、行为效度、临床分型或因果机制。参数合理性的主要证据仍来自公式定义、参数恢复、置换验证、model-free行为锚定和情绪条件敏感性。
""",
        encoding="utf-8",
    )
    SUPPLEMENT.write_text(
        f"""# 统计附录

- 样本量：N=118，T5/T6按`subject_id`一对一匹配。
- 特征数：19；T5为12项，T6为7项。
- 缺失值：0；非有限值：0。
- 标准化：各参数使用样本均值和样本标准差进行z-score。
- 算法：对中心化后的标准化矩阵进行奇异值分解。
- 矩阵秩：{int(summary.iloc[0]['matrix_rank'])}。
- 前五个主成分累计解释方差：{pc5_cumulative:.4f}%。
- PCA为描述性、探索性分析，不进行显著性检验，也不根据载荷事后宣称临床维度。
""",
        encoding="utf-8",
    )
    (BASE / "figure-catalog.md").write_text(
        """# Figure Catalog

## Figure_RawPCA_01_variance_explained

- Purpose: quantify single-component and cumulative explained variance.
- Observation: read exact values from `tables/raw_parameter_pca_summary_N118.csv`.
- Implication: determines how compactly the 19-parameter space can be summarized.
- Caveat: explained variance does not establish parameter validity.

## Figure_RawPCA_02_subject_scores_PC1_PC2

- Purpose: show the continuous participant distribution in the first two components.
- Observation: each point is one of 118 matched participants.
- Implication: describes the current joint parameter space without imposing clusters.
- Caveat: proximity is descriptive and not a clinical grouping result.

## Figure_RawPCA_03_top_loadings_PC1_PC5

- Purpose: identify which current T5/T6 parameters contribute most strongly to PC1-PC5.
- Observation: bars show signed loadings; exact values are in the loading table.
- Implication: supports evidence-bounded interpretation of the joint covariance structure.
- Caveat: component signs are arbitrary and PCA is exploratory.
""",
        encoding="utf-8",
    )


def main() -> None:
    configure_plotting()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_parameters()
    dictionary = feature_dictionary()
    z, checks = zscore(raw)
    coords, loadings, summary = run_pca(z)
    top = top_loadings(loadings, summary)
    raw.to_csv(TABLE_DIR / "raw_parameter_pca_input_N118.csv", index=False, encoding="utf-8-sig")
    z.to_csv(TABLE_DIR / "raw_parameter_pca_zscore_N118.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(TABLE_DIR / "raw_parameter_pca_zscore_quality_checks_N118.csv", index=False, encoding="utf-8-sig")
    coords.to_csv(TABLE_DIR / "raw_parameter_pca_subject_coordinates_N118.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(TABLE_DIR / "raw_parameter_pca_loadings_N118.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "raw_parameter_pca_summary_N118.csv", index=False, encoding="utf-8-sig")
    top.to_csv(TABLE_DIR / "raw_parameter_pca_top_loadings_PC1_PC5_N118.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(TABLE_DIR / "raw_parameter_pca_feature_dictionary_N118.csv", index=False, encoding="utf-8-sig")
    plot_variance(summary)
    plot_scores(coords, summary)
    plot_loadings(top)
    write_strict_bundle(summary, top)
    print(f"Saved strict PCA report: {REPORT}")
    print(f"Saved strict PCA statistics: {SUPPLEMENT}")
    print(f"Saved raw-parameter PCA tables to: {TABLE_DIR}")
    print(f"Saved raw-parameter PCA figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
