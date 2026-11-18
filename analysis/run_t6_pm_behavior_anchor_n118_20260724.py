#!/usr/bin/env python3
"""Anchor current binocular T6 V1 pm estimates to N=118 direct viewing composition."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-t6-pm-anchor-n118-20260724")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/mpl-t6-pm-anchor-n118-20260724")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


OUT = Path(__file__).resolve().parent
REPO = OUT.parents[5]
PARAM = REPO / "t5t6 联合数据分析/260616/data/20260722_t6_118_binocular_rerun/t6_results.csv"
DIRECT = REPO / "t5t6 联合数据分析/260616/任务完成情况整理包_十二大部分_20260616/05_指标和model_free对上/5.9_图表化证明范式捕获注意过程/2026-07-24--N118-Word更新--T6观看份额/tables/2026-07-24--T6-emotion-viewing-proportions-N118.csv"
FIG = OUT / "figures"
TAB = OUT / "tables"
EMOTIONS = ("negative", "positive", "neutral")
COLORS = {"negative": "#D55E00", "positive": "#009E73", "neutral": "#7A7D86"}


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    values = []
    while len(values) < 10_000:
        idx = rng.integers(0, n, size=n)
        if np.unique(x[idx]).size > 1 and np.unique(y[idx]).size > 1:
            result = stats.spearmanr(x[idx], y[idx])
            values.append(float(getattr(result, "statistic", result.correlation)))
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def fdr_bh(values: list[float]) -> list[float]:
    p = np.asarray(values, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1)
    return out.tolist()


def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    params = pd.read_csv(PARAM, dtype={"subject_id": str}, encoding="utf-8-sig").rename(columns={
        "Pm1_positive": "pm_positive",
        "Pm2_negative": "pm_negative",
        "Pm3_neutral": "pm_neutral",
    })
    direct = pd.read_csv(DIRECT, dtype={"subject_id": str}, encoding="utf-8-sig")
    merged = params[["subject_id", "pm_positive", "pm_negative", "pm_neutral"]].merge(
        direct[["subject_id"] + [f"{e}_dwell_time_proportion" for e in EMOTIONS]], on="subject_id", validate="one_to_one"
    )
    if len(merged) != 118:
        raise ValueError(f"Expected N=118, found {len(merged)}")
    if not ((merged[[f"pm_{e}" for e in EMOTIONS]].sum(axis=1) - 1).abs() < 2e-6).all():
        raise ValueError("pm composition does not sum to one")
    rows = []
    for index, emotion in enumerate(EMOTIONS):
        x = merged[f"{emotion}_dwell_time_proportion"].to_numpy(float)
        y = merged[f"pm_{emotion}"].to_numpy(float)
        result = stats.spearmanr(x, y)
        lo, hi = bootstrap_spearman(x, y, 20260724 + index)
        rows.append({"emotion": emotion, "n": 118, "rho": float(getattr(result, "statistic", result.correlation)), "ci95_low": lo, "ci95_high": hi, "p_raw": float(result.pvalue)})
    adjusted = fdr_bh([row["p_raw"] for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["p_fdr_bh"] = p_adj
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for ax, emotion, row in zip(axes, EMOTIONS, rows):
        x = merged[f"{emotion}_dwell_time_proportion"].to_numpy(float)
        y = merged[f"pm_{emotion}"].to_numpy(float)
        ax.scatter(x, y, s=13, alpha=.58, color=COLORS[emotion], edgecolors="none")
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#222222", lw=1)
        ax.set_title(emotion.capitalize())
        ax.set_xlabel("Direct viewing proportion")
        ax.text(.04, .96, f"N = 118\nrho = {row['rho']:.3f}\n95% CI [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]\nFDR p = {row['p_fdr_bh']:.1e}", transform=ax.transAxes, va="top", fontsize=6.5, bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": .92})
        ax.grid(color="#E7E7E7", lw=.5)
    axes[0].set_ylabel("Model-based pm")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(FIG / f"2026-07-24--Figure-T6-pm-viewing-correspondence-N118.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)
    merged.to_csv(TAB / "2026-07-24--T6-pm-direct-viewing-merged-N118.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(TAB / "2026-07-24--T6-pm-direct-viewing-spearman-N118.csv", index=False, encoding="utf-8-sig")
    result_lines = "\n".join(f"- {row['emotion']}: rho={row['rho']:.6f}, bootstrap 95% CI [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}], FDR p={row['p_fdr_bh']:.3e}, N=118。" for row in rows)
    (OUT / "analysis-report.md").write_text(f"""# 2026-07-24 T6 N118 pm 行为锚定

## 分析问题

检验 `pm` 是否表征三类情绪图片在同一试次竞争结构中的相对观看资源构成。

## 结果

{result_lines}

## 解释

三项同情绪对应用于检验 `pm` 的竞争构成含义：某一情绪的 `pm` 越高，其在三张图片之间占据的直接观看份额应越高。

## 证据边界

本结果是经验数据中的行为锚定，不等同于参数恢复。2026-07-24 的 V1 伪似然恢复审计显示：`pm` 绝对误差较小，但正性和负性分量未通过预设的家族级秩恢复阈值，因此不能据此宣称 `pm` 已完成生成式可识别性验证。
""", encoding="utf-8")
    (OUT / "stats-appendix.md").write_text("""# 统计附录

- 分析单位：被试；N=118。
- 参数来源：2026-07-22 双眼重跑的当前 T6 V1 参数表。
- 直接指标：同一批被试、有效双眼 gaze 的三类图片观看份额。
- 相关：Spearman rho；10,000 次被试层 bootstrap 95% CI。
- 多重比较：三个同情绪相关使用 Benjamini–Hochberg FDR 校正。
- 效应量：Spearman rho。
""", encoding="utf-8")
    (OUT / "figure-catalog.md").write_text("""# Figure Catalog

## 2026-07-24--Figure-T6-pm-viewing-correspondence-N118

- Purpose: test whether each emotion-specific pm component tracks the same emotion's share of direct viewing.
- Observation: each panel reports N, rho, bootstrap CI and FDR-adjusted p.
- Implication: supports the empirical interpretation of pm as emotion-image competition composition.
- Caveat: behavioral correspondence does not override the mixed V1 pseudo-likelihood recovery result.
""", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
