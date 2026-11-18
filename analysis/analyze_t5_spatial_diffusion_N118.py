#!/usr/bin/env python3
"""Recompute T5 spatial-diffusion evidence for the N=118 cohort.

The observed metric is subject-level gaze XY dispersion, collapsed with equal
weight across the four T5 emotion conditions. The null baseline uses uniform
random points in normalized image coordinates and matches the number of valid
image points for every subject-emotion row.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-t5-spatial-n118")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/mpl-t5-spatial-n118")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
DATA_DIR = REPO_ROOT / "t5t6 联合数据分析" / "260616" / "data"
DEFAULT_MANIFEST = DATA_DIR / "20260624_t5_t6_118_rerun" / "input_manifest.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "core_model_free_t5_spatial_diffusion_collapsed_N118"
OLD_N80_SUBJECT_EMOTION = (
    SCRIPT_DIR
    / "core_model_free_t5_three_process_N80"
    / "T5_three_process_model_free_subject_long_N80.csv"
)

sys.path.insert(0, str(SCRIPT_DIR))
from rebuild_n80_direct_model_free_from_raw import RawPair, process_t5_pair  # noqa: E402


EMOTIONS = ["negative", "positive", "neutral", "complex"]
BASELINE_SEED = 20260708
BOOTSTRAP_SEED = 20260721


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-subjects", type=int, default=None)
    return parser.parse_args()


def build_pairs(manifest_path: Path, limit_subjects: int | None) -> list[RawPair]:
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str}, encoding="utf-8-sig")
    required = {
        "subject_id",
        "name",
        "task_type",
        "eyetrack_path",
        "records_path",
        "status",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    t5 = manifest[(manifest["task_type"] == "T5") & (manifest["status"] == "success")].copy()
    t5 = t5.sort_values("subject_id").drop_duplicates("subject_id", keep="first")
    if limit_subjects is not None:
        t5 = t5.head(limit_subjects)
    pairs: list[RawPair] = []
    for row in t5.itertuples(index=False):
        records_path = Path(str(row.records_path))
        eyetrack_paths = tuple(Path(item) for item in str(row.eyetrack_path).split(";") if item)
        if not records_path.exists():
            raise FileNotFoundError(f"Missing records file for {row.subject_id}: {records_path}")
        if not eyetrack_paths or not all(path.exists() for path in eyetrack_paths):
            raise FileNotFoundError(f"Missing eyetrack file for {row.subject_id}: {eyetrack_paths}")
        pairs.append(
            RawPair(
                task_type="T5",
                subject=str(row.name),
                subject_id=str(row.subject_id),
                records_path=records_path,
                eyetrack_paths=eyetrack_paths,
                source_rank=0,
                source_label="n118_rerun_manifest",
            )
        )
    return pairs


def random_uniform_dispersion(n_points: int, rng: np.random.Generator) -> float:
    if n_points <= 1:
        return math.nan
    xy = rng.random((n_points, 2))
    return float(np.hypot(np.std(xy[:, 0], ddof=0), np.std(xy[:, 1], ddof=0)))


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        value = float(values[0]) if len(values) else math.nan
        return value, value
    sem = stats.sem(values)
    delta = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    center = float(np.mean(values))
    return center - delta, center + delta


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_resamples: int = 20_000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize(values: np.ndarray, metric: str) -> dict[str, object]:
    clean = values[np.isfinite(values)]
    ci_low, ci_high = mean_ci95(clean)
    return {
        "metric": metric,
        "n_subjects": len(clean),
        "mean": float(np.mean(clean)),
        "sd": float(np.std(clean, ddof=1)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median": float(np.median(clean)),
        "iqr_low": float(np.percentile(clean, 25)),
        "iqr_high": float(np.percentile(clean, 75)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def n80_reproduction_audit(subject_emotion: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "audit",
        "n_compared_rows",
        "n_compared_subjects",
        "max_abs_dispersion_difference",
        "max_abs_valid_points_difference",
        "status",
        "tolerance",
    ]
    if not OLD_N80_SUBJECT_EMOTION.exists():
        return pd.DataFrame(
            [["old_n80_reproduction", 0, 0, math.nan, math.nan, "not_run", "5e-7"]],
            columns=columns,
        )
    old = pd.read_csv(OLD_N80_SUBJECT_EMOTION, dtype={"subject_id": str}, encoding="utf-8-sig")
    old = old[["subject_id", "emotion", "gaze_xy_dispersion", "image_valid_points"]].copy()
    new = subject_emotion[["subject_id", "emotion", "gaze_xy_dispersion", "image_valid_points"]].copy()
    merged = old.merge(new, on=["subject_id", "emotion"], suffixes=("_old", "_new"), how="inner")
    dispersion_diff = np.abs(
        pd.to_numeric(merged["gaze_xy_dispersion_old"], errors="coerce")
        - pd.to_numeric(merged["gaze_xy_dispersion_new"], errors="coerce")
    )
    points_diff = np.abs(
        pd.to_numeric(merged["image_valid_points_old"], errors="coerce")
        - pd.to_numeric(merged["image_valid_points_new"], errors="coerce")
    )
    max_dispersion = float(dispersion_diff.max())
    max_points = float(points_diff.max())
    status = "pass" if len(merged) == 320 and max_dispersion <= 5e-7 and max_points == 0 else "needs_review"
    return pd.DataFrame(
        [[
            "old_n80_reproduction",
            len(merged),
            merged["subject_id"].nunique(),
            max_dispersion,
            max_points,
            status,
            "5e-7",
        ]],
        columns=columns,
    )


def make_figure(subject: pd.DataFrame, comparison: dict[str, object], fig_dir: Path) -> None:
    values = subject["gaze_xy_dispersion"].to_numpy(float)
    observed_mean = float(comparison["observed_mean"])
    observed_low = float(comparison["observed_ci95_low"])
    observed_high = float(comparison["observed_ci95_high"])
    random_mean = float(comparison["random_uniform_mean"])
    random_low = float(comparison["random_uniform_ci95_low"])
    random_high = float(comparison["random_uniform_ci95_high"])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )
    rng = np.random.default_rng(20260629)
    jitter = rng.normal(0, 0.047, len(values))
    fig, ax = plt.subplots(figsize=(3.5, 4.05))
    ax.axhspan(random_low, random_high, color="#8A8F98", alpha=0.15, linewidth=0, zorder=0)
    ax.axhline(random_mean, color="#73777F", lw=1.0, linestyle="--", zorder=1)
    ax.scatter(jitter, values, s=12, color="#4C9F70", alpha=0.52, edgecolors="none", zorder=3)
    ax.errorbar(
        [0],
        [observed_mean],
        yerr=[[observed_mean - observed_low], [observed_high - observed_mean]],
        fmt="o",
        ms=4.5,
        color="#171717",
        ecolor="#171717",
        capsize=3,
        lw=1.1,
        zorder=5,
    )
    ax.axhline(float(np.median(values)), color="#666666", lw=0.75, linestyle=":", zorder=1)
    ax.set_xlim(-0.30, 0.30)
    ax.set_xticks([0])
    ax.set_xticklabels([f"T5 subjects (N={len(values)})"])
    ax.set_ylabel("Gaze XY dispersion")
    ax.set_title("T5 spatial diffusion relative to random baseline", pad=7)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.55)
    ax.text(
        0.285,
        observed_mean,
        f"Observed mean = {observed_mean:.3f}\n95% CI [{observed_low:.3f}, {observed_high:.3f}]",
        ha="right",
        va="center",
        fontsize=6.6,
        color="#171717",
    )
    ax.text(
        0.285,
        random_mean,
        f"Random baseline = {random_mean:.3f}\n95% CI [{random_low:.3f}, {random_high:.3f}]",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#55585E",
    )
    ax.text(
        0.02,
        0.02,
        f"Observed - random = {float(comparison['mean_difference']):.3f}\n"
        f"Bootstrap 95% CI [{float(comparison['difference_bootstrap_ci95_low']):.3f}, "
        f"{float(comparison['difference_bootstrap_ci95_high']):.3f}]\n"
        f"{int(comparison['n_subjects_observed_lower_than_random'])}/{int(comparison['n_subjects'])} lower; "
        f"sign-test p={float(comparison['two_sided_sign_test_p']):.2e}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#4F5258",
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    base = fig_dir / "Figure_T5_gaze_xy_dispersion_collapsed_N118"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=600)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_reports(
    out_dir: Path,
    comparison: dict[str, object],
    audit: pd.DataFrame,
    n_trial_rows: int,
    n_valid_trial_rows: int,
    n_subject_emotion_rows: int,
) -> None:
    audit_status = str(audit.iloc[0]["status"])
    report = f"""# T5 空间扩散 N118 分析报告

## 分析问题

T5 连续 gaze 轨迹是否表现为低于匹配随机均匀观看的空间扩散程度。

## 数据与分析单位

- 被试：{int(comparison['n_subjects'])} 人。
- 原始试次行：{n_trial_rows}；有效 trigger 窗口试次：{n_valid_trial_rows}。
- 被试×情绪聚合行：{n_subject_emotion_rows}。
- 被试层指标：先在每个情绪条件内平均试次 `gaze_xy_dispersion`，再对四类情绪等权平均。
- 随机基准：在 `[0,1] × [0,1]` 归一化图片坐标中生成均匀随机点，并逐被试×情绪匹配有效 gaze 点数。

## 主要发现

- 真实 gaze 空间扩散均值为 {float(comparison['observed_mean']):.6f}，95% CI [{float(comparison['observed_ci95_low']):.6f}, {float(comparison['observed_ci95_high']):.6f}]。
- 随机均匀基准均值为 {float(comparison['random_uniform_mean']):.6f}，95% CI [{float(comparison['random_uniform_ci95_low']):.6f}, {float(comparison['random_uniform_ci95_high']):.6f}]。
- 配对均值差为 {float(comparison['mean_difference']):.6f}，bootstrap 95% CI [{float(comparison['difference_bootstrap_ci95_low']):.6f}, {float(comparison['difference_bootstrap_ci95_high']):.6f}]。
- {int(comparison['n_subjects_observed_lower_than_random'])}/{int(comparison['n_subjects'])} 名被试低于随机基准；双侧精确 sign test p={float(comparison['two_sided_sign_test_p']):.3e}。

## 解释边界

结果支持 T5 范式能够从原始 gaze 轨迹中捕捉非随机的空间集中结构。该比较没有区分情绪条件，也不支持临床诊断差异或抑郁症机制结论。随机均匀基准只匹配有效点数，没有保留图片显著性、边缘限制、时间相关或缺失模式。

## 复现检查

原 N80 子集复现状态：`{audit_status}`。详细误差见 `T5_spatial_diffusion_N80_reproduction_audit.csv`。
"""
    stats_text = f"""# T5 空间扩散 N118 统计附录

- 分析单位：被试。
- N：{int(comparison['n_subjects'])}。
- 中心与离散：均值、样本标准差、中位数和 IQR。
- 均值区间：t 分布 95% CI。
- 主要配对差异区间：20,000 次被试层 bootstrap 均值差 95% CI，随机种子 {BOOTSTRAP_SEED}。
- 推断检验：双侧精确 sign test；不要求配对差服从正态分布。
- 效应量：paired Cohen dz={float(comparison['paired_cohen_dz']):.6f}。
- 多重比较：无；本任务只有一个预定义主比较。
- 随机基准种子：{BASELINE_SEED}。
- 配对差 Shapiro-Wilk：W={float(comparison['difference_shapiro_w']):.6f}, p={float(comparison['difference_shapiro_p']):.6g}；该结果仅用于分布审计，不改变 sign test 选择。
"""
    catalog = f"""# Figure Catalog

## Figure_T5_gaze_xy_dispersion_collapsed_N118

- Purpose: show whether observed T5 gaze is spatially more concentrated than a point-count-matched uniform baseline.
- Data source: `T5_gaze_xy_dispersion_subject_N118.csv`.
- Unit: subject, N={int(comparison['n_subjects'])}.
- Center/interval: black point and error bar show observed mean and t-based 95% CI; grey dashed line and band show the random-baseline mean and 95% CI.
- Key observation: {int(comparison['n_subjects_observed_lower_than_random'])}/{int(comparison['n_subjects'])} subjects are below the matched random baseline.
- Interpretation: the T5 task captures non-random spatial concentration in continuous gaze trajectories.
- Caveat: this is a paradigm-level spatial null comparison, not an emotion-condition or clinical-group comparison.
- Exports: PNG (600 dpi), PDF, SVG with editable text.
"""
    (out_dir / "analysis-report.md").write_text(report, encoding="utf-8")
    (out_dir / "stats-appendix.md").write_text(stats_text, encoding="utf-8")
    (out_dir / "figure-catalog.md").write_text(catalog, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.manifest, args.limit_subjects)
    trial_rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs, start=1):
        trial_rows.extend(process_t5_pair(pair))
        if index % 10 == 0 or index == len(pairs):
            print(f"processed_subjects={index}/{len(pairs)}", flush=True)
    trial = pd.DataFrame(trial_rows)
    trial["subject_id"] = trial["subject_id"].astype(str)
    for column in ["image_valid_points", "gaze_xy_dispersion"]:
        trial[column] = pd.to_numeric(trial[column], errors="coerce")
    valid = trial[(trial["cut_status"] == "cut") & trial["gaze_xy_dispersion"].notna()].copy()
    valid["emotion"] = pd.Categorical(valid["emotion"], categories=EMOTIONS, ordered=True)
    subject_emotion = (
        valid.groupby(["subject", "subject_id", "emotion"], observed=True, as_index=False)
        .agg(
            valid_trial_count=("trial", "count"),
            image_valid_points=("image_valid_points", "sum"),
            gaze_xy_dispersion=("gaze_xy_dispersion", "mean"),
        )
        .sort_values(["subject_id", "emotion"])
    )
    baseline_rng = np.random.default_rng(BASELINE_SEED)
    subject_emotion["random_uniform_gaze_xy_dispersion"] = [
        random_uniform_dispersion(int(n), baseline_rng)
        for n in subject_emotion["image_valid_points"].to_numpy(float)
    ]
    subject = (
        subject_emotion.groupby(["subject", "subject_id"], as_index=False)
        .agg(
            valid_condition_count=("emotion", "count"),
            valid_trial_count=("valid_trial_count", "sum"),
            image_valid_points=("image_valid_points", "sum"),
            gaze_xy_dispersion=("gaze_xy_dispersion", "mean"),
            random_uniform_gaze_xy_dispersion=("random_uniform_gaze_xy_dispersion", "mean"),
        )
        .sort_values("gaze_xy_dispersion")
    )
    subject["observed_minus_random_uniform"] = (
        subject["gaze_xy_dispersion"] - subject["random_uniform_gaze_xy_dispersion"]
    )

    observed = subject["gaze_xy_dispersion"].to_numpy(float)
    random_values = subject["random_uniform_gaze_xy_dispersion"].to_numpy(float)
    differences = subject["observed_minus_random_uniform"].to_numpy(float)
    observed_ci = mean_ci95(observed)
    random_ci = mean_ci95(random_values)
    bootstrap_ci = bootstrap_mean_ci(differences, BOOTSTRAP_SEED)
    n_lower = int(np.sum(differences < 0))
    sign_result = stats.binomtest(n_lower, len(differences), p=0.5, alternative="two-sided")
    shapiro = stats.shapiro(differences) if len(differences) >= 3 else None
    comparison = {
        "comparison": "observed_minus_random_uniform_baseline",
        "n_subjects": len(subject),
        "n_subjects_with_4_conditions": int((subject["valid_condition_count"] == 4).sum()),
        "observed_mean": float(np.mean(observed)),
        "observed_ci95_low": observed_ci[0],
        "observed_ci95_high": observed_ci[1],
        "random_uniform_mean": float(np.mean(random_values)),
        "random_uniform_ci95_low": random_ci[0],
        "random_uniform_ci95_high": random_ci[1],
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "difference_bootstrap_ci95_low": bootstrap_ci[0],
        "difference_bootstrap_ci95_high": bootstrap_ci[1],
        "paired_cohen_dz": float(np.mean(differences) / np.std(differences, ddof=1)),
        "n_subjects_observed_lower_than_random": n_lower,
        "two_sided_sign_test_p": float(sign_result.pvalue),
        "difference_shapiro_w": float(shapiro.statistic) if shapiro is not None else math.nan,
        "difference_shapiro_p": float(shapiro.pvalue) if shapiro is not None else math.nan,
        "baseline_seed": BASELINE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "baseline_definition": "Uniform random points in normalized image coordinates [0,1] x [0,1], matched per subject-emotion to image_valid_points.",
    }
    summary = pd.DataFrame(
        [
            summarize(observed, "observed_gaze_xy_dispersion_collapsed_across_emotions"),
            summarize(random_values, "random_uniform_baseline_matched_points"),
            summarize(differences, "observed_minus_random_uniform_baseline"),
        ]
    )
    audit = n80_reproduction_audit(subject_emotion)

    trial.to_csv(args.output_dir / "T5_gaze_xy_dispersion_trial_N118.csv", index=False, encoding="utf-8-sig")
    subject_emotion.to_csv(
        args.output_dir / "T5_gaze_xy_dispersion_subject_emotion_N118.csv", index=False, encoding="utf-8-sig"
    )
    subject.to_csv(args.output_dir / "T5_gaze_xy_dispersion_subject_N118.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "T5_gaze_xy_dispersion_summary_N118.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([comparison]).to_csv(
        args.output_dir / "T5_gaze_xy_dispersion_random_baseline_comparison_N118.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit.to_csv(
        args.output_dir / "T5_spatial_diffusion_N80_reproduction_audit.csv", index=False, encoding="utf-8-sig"
    )
    make_figure(subject, comparison, args.output_dir / "figures")
    write_reports(
        args.output_dir,
        comparison,
        audit,
        n_trial_rows=len(trial),
        n_valid_trial_rows=len(valid),
        n_subject_emotion_rows=len(subject_emotion),
    )
    print(pd.DataFrame([comparison]).to_string(index=False))
    print(audit.to_string(index=False))
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
