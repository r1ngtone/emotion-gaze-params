#!/usr/bin/env python3
"""Build the current-only Step 7 strict analysis bundle."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-step7-valid-n118-20260729")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/mpl-step7-valid-n118-20260729")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
CONDITION_SOURCE = ROOT / "data/20260727_t5_118_valid_rerun/core_analysis_comparison/condition-contrasts-new.csv"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
CONDITION_OUT = TABLES / "2026-07-29--T5-valid-condition-contrasts-N118.csv"
CROSS_BOOT = TABLES / "t5_t6_cross_task_parameter_correlations_bootstrap_N118.csv"
EMOTION_LABELS = {
    "positive-negative": "Positive - Negative",
    "positive-neutral": "Positive - Neutral",
    "positive-complex": "Positive - Complex",
    "negative-neutral": "Negative - Neutral",
    "negative-complex": "Negative - Complex",
    "neutral-complex": "Neutral - Complex",
}


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def plot_conditions(contrasts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.4), sharey=True, constrained_layout=True)
    for ax, metric in zip(axes, ["T_half", "H_bar_norm", "I_focus"]):
        sub = contrasts.loc[contrasts["metric"].eq(metric)].copy()
        sub["label"] = sub["contrast"].map(EMOTION_LABELS)
        sub = sub.iloc[::-1]
        y = np.arange(len(sub))
        significant = sub["q_bh_all_18"].to_numpy(float) < 0.05
        colors = np.where(significant, "#0072B2", "#999999")
        ax.hlines(y, sub["ci_low"], sub["ci_high"], color=colors, lw=1.6)
        ax.scatter(sub["mean_difference"], y, color=colors, s=28, zorder=3)
        ax.axvline(0, color="#222222", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"])
        ax.set_xlabel("Mean difference (95% CI)")
        ax.set_title(metric)
        ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    save(fig, "Figure_01_T5_valid_condition_contrasts_N118")


def significant_lines(contrasts: pd.DataFrame) -> str:
    selected = contrasts.loc[contrasts["q_bh_all_18"].lt(0.05)].copy()
    return "\n".join(
        f"- `{row.metric}`，{row.contrast}：差值={row.mean_difference:.4f}，95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}]，dz={row.paired_dz:.3f}，BH-FDR q={row.q_bh_all_18:.4g}。"
        for row in selected.itertuples(index=False)
    )


def cross_lines(cross: pd.DataFrame) -> str:
    top = cross.assign(abs_rho=lambda d: d["spearman_rho"].abs()).sort_values("abs_rho", ascending=False).head(8)
    return "\n".join(
        f"- `{row.parameter_a}`—`{row.parameter_b}`：rho={row.spearman_rho:.3f}，bootstrap 95% CI [{row.rho_ci_low:.3f}, {row.rho_ci_high:.3f}]。"
        for row in top.itertuples(index=False)
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONDITION_SOURCE, CONDITION_OUT)
    contrasts = pd.read_csv(CONDITION_OUT, encoding="utf-8-sig")
    cross = pd.read_csv(CROSS_BOOT, encoding="utf-8-sig")
    if len(contrasts) != 18 or set(contrasts["n"]) != {118}:
        raise ValueError("Expected 18 complete N=118 T5 condition contrasts")
    if set(cross["n"]) != {118} or cross[["spearman_rho", "rho_ci_low", "rho_ci_high"]].isna().any().any():
        raise ValueError("Cross-task bootstrap table failed N=118 or finite-value QA")
    plot_conditions(contrasts)
    n_sig = int(contrasts["q_bh_all_18"].lt(0.05).sum())
    (BASE / "analysis-report.md").write_text(
        f"""# T5 valid-gaze N=118条件效应与T5–T6参数关系

生成日期：2026-07-29

## 分析问题

本分析回答两个问题：第一，当前T5正式参数是否对预先定义的图片条件差异敏感；第二，当前T5与T6参数在同一批参与者中呈现哪些跨任务协变关系。

## 固定输入

- T5：`20260727_t5_118_valid_rerun/t5_results.csv`，样本层面保留`valid == 1`，N=118。
- T6：`20260722_t6_118_binocular_rerun/t6_results.csv`，有效双眼，N=118。
- 分析单位：参与者；两任务按`subject_id`一对一匹配。

## T5条件对比

18项预定义配对对比中，{n_sig}项在全体18项BH-FDR校正后达到q<0.05：

{significant_lines(contrasts)}

`T_half`的六项条件对比均未通过FDR校正，因此当前结果不支持将其作为稳定的情绪条件效应。`H_bar_norm`的负性—正性结构仍是T5最强的条件证据；`I_focus`的负性—正性和负性—中性差异作为次级证据。

![T5 condition contrasts](figures/Figure_01_T5_valid_condition_contrasts_N118.png)

图1回答当前T5参数对哪些图片条件差异敏感。点为配对均值差，线为bootstrap 95% CI；蓝色表示18项BH-FDR校正后q<0.05，灰色表示未通过校正。

## T5–T6跨任务关系

绝对Spearman相关最高的八项及其参与者bootstrap区间为：

{cross_lines(cross)}

这些关系用于描述两项平行任务之间的协变结构。即使区间不跨0，也不应表述为一个任务验证另一个任务，或表述为稳定个体特质。

![Cross-task correlations](figures/Figure_06_T5_T6_direct_bootstrap_ci_top_correlations.png)

图2回答哪些直接T5条件参数与T6参数具有较强的跨任务秩相关，以及这些相关在参与者bootstrap中的不确定性。相关是探索性结构证据，不承担参数定义或参数恢复职责。

## 结论边界

条件敏感性可以作为参数合理性的经验收敛证据之一；跨任务相关可以说明两项任务存在互补协变，但二者均不能替代参数恢复、likelihood置换或model-free行为锚定。
""",
        encoding="utf-8",
    )
    (BASE / "stats-appendix.md").write_text(
        """# 统计附录

## T5条件对比

- 分析单位：参与者，N=118，配对设计。
- 对比数：3个参数家族各6项，共18项预定义条件对比。
- 区间：参与者层bootstrap 95% CI。
- 检验：双侧配对符号翻转检验。
- 效应量：配对Cohen's dz。
- 多重比较：18项统一使用Benjamini–Hochberg FDR校正。

## 参数关系

- 相关：Spearman rho。
- 区间：2,000次参与者bootstrap 95% CI。
- 跨任务相关作为探索性结构描述；报告完整结果，不将未校正相关解释为验证性发现。
- T6三个pm存在精确组成约束，因此不把三项pm同时进入普通多元回归或按独立变量解释VIF。
""",
        encoding="utf-8",
    )
    (BASE / "figure-catalog.md").write_text(
        """# Figure Catalog

## Figure_01_T5_valid_condition_contrasts_N118

- Purpose: show all 18 current T5 paired condition contrasts without selecting only significant results.
- Data: current T5 valid-gaze N=118 condition-contrast table.
- Observation: point estimates, bootstrap 95% CIs and FDR status are visible for every contrast.
- Implication: identifies which theoretically relevant condition differences remain supported after validity filtering.
- Caveat: condition sensitivity is convergent evidence, not parameter recovery.

## Figure_06_T5_T6_direct_bootstrap_ci_top_correlations

- Purpose: summarize the largest current cross-task parameter correlations and their uncertainty.
- Data: matched T5 valid-gaze and T6 valid-binocular N=118 parameters.
- Observation: each interval is based on 2,000 participant bootstrap samples.
- Implication: describes complementary covariance between the parallel tasks.
- Caveat: displayed correlations are exploratory and do not establish stable traits or causal mechanisms.
""",
        encoding="utf-8",
    )
    print(f"condition_contrasts={len(contrasts)} fdr_significant={n_sig} cross_task_rows={len(cross)}")


if __name__ == "__main__":
    main()
