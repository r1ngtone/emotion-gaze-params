#!/usr/bin/env python3
"""Associate current 2026-07-29 T5-valid/T6-binocular PCA scores with scale totals."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
SECTION_10_3 = BASE.parent
BUNDLE_ROOT = BASE.parents[2]
SOURCE = (
    SECTION_10_3
    / "N118_RawPCA与量表总分关联"
    / "analyze_raw_pca_scale_total_correlations_N118.py"
)
PCA_BUNDLE = (
    BUNDLE_ROOT
    / "08_每人做成向量空间"
    / "2026-07-29--T5-valid-N118--联合参数PCA"
)
COORDS = PCA_BUNDLE / "tables" / "raw_parameter_pca_subject_coordinates_N118.csv"
PCA_SUMMARY = PCA_BUNDLE / "tables" / "raw_parameter_pca_summary_N118.csv"
PCA_TOP_LOADINGS = PCA_BUNDLE / "tables" / "raw_parameter_pca_top_loadings_PC1_PC5_N118.csv"

TABLE_DIR = BASE / "tables"
FIG_DIR = BASE / "figures"
REPORT = BASE / "analysis-report.md"
STATS_APPENDIX = BASE / "stats-appendix.md"
FIGURE_CATALOG = BASE / "figure-catalog.md"

PC_INTERPRETATIONS = {
    "PC1": "T5 cross-condition gaze-history decay timescale (T_half-dominant)",
    "PC2": "T5 spatial organization with positive-channel peak timing",
    "PC3": "T6 negative-positive resource and peak-time competition",
    "PC4": "T6 neutral-positive resource and peak-time competition",
    "PC5": "T6 peak timing/spatial scale with a T5 neutral-history contribution",
}


def load_workflow_module():
    spec = importlib.util.spec_from_file_location("scale_total_workflow", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source workflow: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_module(module) -> None:
    module.BASE = BASE
    module.COORDS_FILE = COORDS
    module.TABLE_DIR = TABLE_DIR
    module.FIG_DIR = FIG_DIR
    module.REPORT_FILE = REPORT
    module.STATS_APPENDIX_FILE = STATS_APPENDIX
    module.FIGURE_CATALOG_FILE = FIGURE_CATALOG
    module.PC_INTERPRETATIONS = PC_INTERPRETATIONS
    module.N_BOOT = 5_000
    module.BOOTSTRAP_SEED = 20260729


def save_current_context(module) -> None:
    coords, totals = module.load_inputs()
    merged = coords.merge(totals, on="subject_id", how="inner", validate="one_to_one")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TABLE_DIR / "current_pca_scale_matched_input_N80.csv", index=False, encoding="utf-8-sig")

    summary = pd.read_csv(PCA_SUMMARY, encoding="utf-8-sig")
    top = pd.read_csv(PCA_TOP_LOADINGS, encoding="utf-8-sig")
    summary.loc[summary["component"].isin(module.PCS[:5])].to_csv(
        TABLE_DIR / "current_pca_summary_PC1_PC5_N118.csv", index=False, encoding="utf-8-sig"
    )
    top.loc[top["component"].isin(module.PCS[:5])].to_csv(
        TABLE_DIR / "current_pca_top_loadings_PC1_PC5_N118.csv", index=False, encoding="utf-8-sig"
    )


def plot_all_interval_estimates(correlations: pd.DataFrame) -> str:
    ordered = correlations.copy()
    ordered["label"] = ordered["pc"] + " — " + ordered["scale"]
    ordered["pc_order"] = ordered["pc"].str.extract(r"(\d+)").astype(int)
    ordered = ordered.sort_values(["pc_order", "scale"], ascending=[False, True]).reset_index(drop=True)

    y = np.arange(len(ordered))
    robust = ordered["robust_exploratory_association"].to_numpy(dtype=bool)
    colors = np.where(robust, "#D55E00", "#0072B2")
    left = ordered["rho"] - ordered["ci95_low"]
    right = ordered["ci95_high"] - ordered["rho"]

    fig, ax = plt.subplots(figsize=(8.2, 10.2), constrained_layout=True)
    ax.axvline(0, color="#444444", lw=1.0, ls="--", zorder=0)
    for idx in range(len(ordered)):
        ax.errorbar(
            ordered.loc[idx, "rho"],
            y[idx],
            xerr=np.array([[left.iloc[idx]], [right.iloc[idx]]]),
            fmt="o",
            color=colors[idx],
            ecolor=colors[idx],
            markersize=4.5,
            capsize=2.0,
            elinewidth=1.1,
        )
    ax.set_yticks(y, ordered["label"], fontsize=7.5)
    ax.set_xlabel("Spearman rho with paired-bootstrap 95% CI")
    ax.set_xlim(-0.55, 0.55)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.text(
        0.99,
        0.01,
        "Orange: global BH-FDR q < .05 and CI excludes 0",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
    )
    stem = "Figure_CurrentPCA_ScaleTotal_02_all_effect_intervals"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return stem


def refresh_report(correlations: pd.DataFrame, interval_stem: str) -> None:
    text = REPORT.read_text(encoding="utf-8")
    text = text.replace("生成日期：2026-07-11", "生成日期：2026-07-29")
    text = text.replace(
        "旧的 joint-vector PCA 坐标未参与本轮计算；本轮只读取 raw-PCA 的 N=118 被试坐标",
        "旧PCA坐标未参与本轮计算；本轮只读取2026年7月29日T5-valid/T6有效双眼联合PCA的N=118被试坐标",
    )
    text = text.replace(
        "- 脚本：`analyze_raw_pca_scale_total_correlations_N118.py`",
        "- 脚本：`run_current_pca_scale_total_20260729.py`",
    )
    robust = correlations.loc[correlations["robust_exploratory_association"]]
    largest = correlations.sort_values("abs_rho", ascending=False).head(5)
    interpretation = [
        "",
        "## 7. 逐图解释与第三主要成果定位",
        "",
        "### 图1：PC—量表总分相关热图",
        "",
        "**回答的问题。** 当前联合PCA的五个低维被试表征，是否与六个量表总分形成经全局多重比较校正后仍可保留的外部表型关联？",
        "",
        "**图中依据。** 颜色和单元格数字表示全部30项Spearman相关；星号只标记全局BH-FDR `q<.05`的结果。热图完整呈现所有分析，不依据显著性筛选单元格。",
        "",
    ]
    if robust.empty:
        interpretation.extend(
            [
                "**观察。** 当前没有任何PC—量表总分关系同时满足全局BH-FDR和bootstrap区间不跨0的标准。",
                "",
                "**影响。** PCA仍完成了把19个参数压缩为低维联合表征的预定任务，但当前N=69–70量表子样本没有提供稳健的外部表型关联。第三主要成果应把这一部分报告为探索性阴性结果，而不是诊断或临床效度证据。",
            ]
        )
    else:
        interpretation.extend(
            [
                f"**观察。** 30项关系中有{len(robust)}项同时满足全局BH-FDR和bootstrap区间不跨0。",
                "",
                "**影响。** 这些关系可作为探索性外部表型关联进入第三主要成果末尾，但仍需量表版本核对和独立样本复现。",
            ]
        )
    interpretation.extend(
        [
            "",
            "### 图2：全部30项相关的效应量与不确定性",
            "",
            f"![All PC–scale effect intervals](figures/{interval_stem}.png)",
            "",
            "**回答的问题。** 在不只查看p值的情况下，30项相关的方向、大小和估计不确定性如何分布？",
            "",
            "**图中依据。** 每个点为Spearman rho，水平线为5,000次配对bootstrap 95% CI，虚线为零相关；橙色仅表示同时满足全局BH-FDR和区间不跨0。",
            "",
            "**观察。** 绝对效应最大的五项为：",
            "",
        ]
    )
    for _, row in largest.iterrows():
        interpretation.append(
            f"- `{row['pc']}`—{row['scale_total_column']}：rho={row['rho']:.3f}，"
            f"95% CI [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]，"
            f"q={row['q_value_bh_global']:.3f}，N={int(row['n_valid'])}。"
        )
    interpretation.extend(
        [
            "",
            "**影响。** 效应图用于区分‘估计值存在一定大小’与‘经多重比较后获得稳定支持’。未经校正的较大估计值只能用于后续假设生成，不能作为当前确认性结论。",
            "",
            "### 与第三主要成果的关系",
            "",
            "本分析位于第三主要成果的末端：情绪条件敏感性说明部分参数能够区分实验条件；联合PCA把19个参数压缩为PC1–PC5；本分析再检验这些低维表征是否连接到外部量表。量表关联不承担参数恢复或行为锚定证明，也不作为临床诊断效度。",
            "",
            "## 8. 当前证据边界",
            "",
            "- PCA在全部118名参与者中拟合；量表关联只使用按`subject_id`匹配后各量表均非缺失的N=69–70。",
            "- 六个量表总分来自现存预计算字段；正式论文解释方向前仍需核对正式版本、反向题和高分含义。",
            "- PC符号可整体翻转，相关方向必须结合当前载荷表阅读。",
            "- 30项总分相关构成一个预先固定的多重比较族；不使用题项级未校正结果补充或替代总分结论。",
            "- 当前分析为非临床样本中的探索性外部关联，不支持因果关系、诊断效度或临床分型。",
            "",
            "## 9. 当前PCA来源与附加复核表",
            "",
            "- 当前PC坐标：`tables/current_pca_scale_matched_input_N80.csv`中的PC1–PC5列，来源为7月29日N=118联合PCA。",
            "- 当前PCA解释方差：`tables/current_pca_summary_PC1_PC5_N118.csv`。",
            "- 当前PCA主要载荷：`tables/current_pca_top_loadings_PC1_PC5_N118.csv`。",
            "- 全部30项相关：`tables/raw_pca_scale_total_spearman_correlations_N118.csv`。",
        ]
    )
    REPORT.write_text(text.rstrip() + "\n" + "\n".join(interpretation) + "\n", encoding="utf-8")


def refresh_figure_catalog(interval_stem: str) -> None:
    with FIGURE_CATALOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\n## "
            + interval_stem
            + "\n\n"
            + "- Purpose: show all 30 effect estimates and paired-bootstrap 95% intervals without significance-only selection.\n"
            + "- Data: current 2026-07-29 PC1-PC5 scores matched to six scale totals.\n"
            + "- Encoding: points are Spearman rho; horizontal bars are 5,000-resample paired-bootstrap 95% CIs; orange marks global BH-FDR q < .05 with CI excluding zero.\n"
            + "- Reader should notice: effect magnitude and uncertainty can differ from uncorrected p-value ranking.\n"
            + "- Decision impact: only orange estimates may be described as robust exploratory associations; all others remain hypothesis-generating.\n"
            + "- Caveat: scale versions and score directions still require formal metadata confirmation.\n"
        )


def main() -> None:
    module = load_workflow_module()
    configure_module(module)
    module.main()
    save_current_context(module)
    correlation_file = TABLE_DIR / "raw_pca_scale_total_spearman_correlations_N118.csv"
    correlations = pd.read_csv(correlation_file, encoding="utf-8-sig")
    if len(correlations) != 30:
        raise AssertionError(f"Expected 30 correlations, found {len(correlations)}")
    interval_stem = plot_all_interval_estimates(correlations)
    refresh_report(correlations, interval_stem)
    refresh_figure_catalog(interval_stem)
    print(f"Current PCA source: {COORDS}")
    print(f"Matched PC-scale rows: {len(pd.read_csv(TABLE_DIR / 'current_pca_scale_matched_input_N80.csv', encoding='utf-8-sig'))}")
    print(f"Robust exploratory associations: {int(correlations['robust_exploratory_association'].sum())}/30")


if __name__ == "__main__":
    main()
