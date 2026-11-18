#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-param-rel-valid-n118-20260729")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
DATA = ROOT / "data"
T5_FILE = ROOT / "data" / "20260727_t5_118_valid_rerun" / "t5_results.csv"
T6_FILE = ROOT / "data" / "20260722_t6_118_binocular_rerun" / "t6_results.csv"

FIG_DIR = BASE / "figures"
TAB_DIR = BASE / "tables"
REPORT = BASE / "2026-07-29--T5-valid-N118参数间关系与共线性分析报告.md"

IMAGE_TYPES = ["positive", "negative", "neutral", "complex"]
T5_PARAM_CANDIDATES = ["T_half", "H_bar_norm", "I_focus"]
T6_RENAME = {
    "Pm1_positive": "pm_positive",
    "Pm2_negative": "pm_negative",
    "Pm3_neutral": "pm_neutral",
    "Tmax1": "Tmax_positive",
    "Tmax2": "Tmax_negative",
    "Tmax3": "Tmax_neutral",
    "类别": "predicted_class_t6",
}
T6_COLS = [
    "pm_positive",
    "pm_negative",
    "pm_neutral",
    "Tmax_positive",
    "Tmax_negative",
    "Tmax_neutral",
    "dc",
]
N_BOOT = 2000
BOOT_SEED = 20260705
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99
LOO_STABLE_DELTA = 0.10


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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    t5 = pd.read_csv(T5_FILE, encoding="utf-8-sig")
    t6 = pd.read_csv(T6_FILE, encoding="utf-8-sig").rename(columns=T6_RENAME)
    t5["subject_id"] = t5["subject_id"].astype(str)
    t6["subject_id"] = t6["subject_id"].astype(str)
    if len(t5) != 118 or len(t6) != 118:
        raise ValueError(f"Expected N=118 for both tasks, got T5={len(t5)}, T6={len(t6)}")
    if len(t5[["subject_id"]].merge(t6[["subject_id"]], on="subject_id")) != 118:
        raise ValueError("T5/T6 subject_id inner join is not N=118")

    t5_cols = [f"{img}_{param}" for img in IMAGE_TYPES for param in T5_PARAM_CANDIDATES if f"{img}_{param}" in t5.columns]
    missing_t5 = [f"{img}_{param}" for img in IMAGE_TYPES for param in T5_PARAM_CANDIDATES if f"{img}_{param}" not in t5.columns]
    missing_t6 = [col for col in T6_COLS if col not in t6.columns]
    if missing_t6:
        raise ValueError(f"Missing T6 columns: {missing_t6}")
    if not t5_cols:
        raise ValueError("No expected T5 parameter columns found")

    for label, df, cols in [("T5", t5, t5_cols), ("T6", t6, T6_COLS)]:
        values = df[cols].to_numpy(dtype=float)
        if np.isnan(values).any() or not np.isfinite(values).all():
            raise ValueError(f"{label} contains missing or non-finite parameter values")

    if missing_t5:
        print("T5 missing candidate columns:", ", ".join(missing_t5))
    return t5, t6, t5_cols, T6_COLS


def zscore(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype=float)
    sd = x.std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError(f"Zero-variance columns: {[col for col, val in zip(cols, sd) if val == 0]}")
    return (x - x.mean(axis=0)) / sd


def bh_fdr(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out.tolist()
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[valid])]
    ranked = p[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.clip(adj, 0, 1)
    return out.tolist()


def pairwise_correlations(df: pd.DataFrame, cols: list[str], family: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            valid = df[[a, b]].dropna()
            pearson = valid[a].corr(valid[b], method="pearson")
            res = spearmanr(valid[a], valid[b])
            rho = float(getattr(res, "statistic", res.correlation))
            rows.append(
                {
                    "family": family,
                    "parameter_a": a,
                    "parameter_b": b,
                    "n": int(len(valid)),
                    "pearson_r": float(pearson),
                    "spearman_rho": rho,
                    "spearman_p": float(res.pvalue),
                    "abs_spearman_rho": abs(rho),
                    "abs_pearson_r": abs(float(pearson)),
                }
            )
    out = pd.DataFrame(rows)
    out["spearman_p_fdr_bh"] = bh_fdr(out["spearman_p"].tolist())
    return out.sort_values(["abs_spearman_rho", "abs_pearson_r"], ascending=[False, False])


def matrix_diagnostics(df: pd.DataFrame, cols: list[str], family: str) -> dict[str, object]:
    x = zscore(df, cols)
    s = np.linalg.svd(x, compute_uv=False)
    rank = int(np.linalg.matrix_rank(x, tol=1e-10))
    condition = np.inf if s[-1] < 1e-12 else float(s[0] / s[-1])
    return {
        "family": family,
        "n": int(len(df)),
        "n_parameters": int(len(cols)),
        "matrix_rank": rank,
        "rank_deficit": int(len(cols) - rank),
        "condition_number": condition,
        "smallest_singular_value": float(s[-1]),
    }


def vif_table(df: pd.DataFrame, cols: list[str], family: str) -> pd.DataFrame:
    x = zscore(df, cols)
    rows: list[dict[str, object]] = []
    for i, col in enumerate(cols):
        y = x[:, i]
        others = np.delete(x, i, axis=1)
        coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        fitted = others @ coef
        ss_res = float(((y - fitted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
        vif = np.inf if r2 >= 1 - 1e-10 else 1.0 / (1.0 - r2)
        rows.append({"family": family, "parameter": col, "R2_from_other_parameters": r2, "VIF": vif})
    return pd.DataFrame(rows).sort_values(["VIF", "R2_from_other_parameters"], ascending=False)


def collapsed_t5(t5: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame({"subject_id": t5["subject_id"]})
    cols: list[str] = []
    for param in T5_PARAM_CANDIDATES:
        source = [f"{img}_{param}" for img in IMAGE_TYPES if f"{img}_{param}" in t5.columns]
        if source:
            col = f"T5_mean_{param}"
            out[col] = t5[source].mean(axis=1)
            cols.append(col)
    return out, cols


def cross_task_correlations(t5: pd.DataFrame, t6: pd.DataFrame) -> pd.DataFrame:
    t5_mean, t5_mean_cols = collapsed_t5(t5)
    merged = t5_mean.merge(t6[["subject_id"] + T6_COLS], on="subject_id", how="inner")
    rows: list[dict[str, object]] = []
    for a in t5_mean_cols:
        for b in T6_COLS:
            valid = merged[[a, b]].dropna()
            pearson = valid[a].corr(valid[b], method="pearson")
            res = spearmanr(valid[a], valid[b])
            rho = float(getattr(res, "statistic", res.correlation))
            rows.append(
                {
                    "family": "T5_T6_cross_task",
                    "parameter_a": a,
                    "parameter_b": b,
                    "n": int(len(valid)),
                    "pearson_r": float(pearson),
                    "spearman_rho": rho,
                    "spearman_p": float(res.pvalue),
                    "abs_spearman_rho": abs(rho),
                    "abs_pearson_r": abs(float(pearson)),
                }
            )
    out = pd.DataFrame(rows)
    out["spearman_p_fdr_bh"] = bh_fdr(out["spearman_p"].tolist())
    return out.sort_values(["abs_spearman_rho", "abs_pearson_r"], ascending=[False, False])


def direct_cross_task_correlations(t5: pd.DataFrame, t6: pd.DataFrame, t5_cols: list[str]) -> pd.DataFrame:
    merged = t5[["subject_id"] + t5_cols].merge(t6[["subject_id"] + T6_COLS], on="subject_id", how="inner")
    rows: list[dict[str, object]] = []
    for a in t5_cols:
        for b in T6_COLS:
            valid = merged[[a, b]].dropna()
            pearson = valid[a].corr(valid[b], method="pearson")
            res = spearmanr(valid[a], valid[b])
            rho = float(getattr(res, "statistic", res.correlation))
            rows.append(
                {
                    "family": "T5_T6_direct_cross_task",
                    "parameter_a": a,
                    "parameter_b": b,
                    "n": int(len(valid)),
                    "pearson_r": float(pearson),
                    "spearman_rho": rho,
                    "spearman_p": float(res.pvalue),
                    "abs_spearman_rho": abs(rho),
                    "abs_pearson_r": abs(float(pearson)),
                }
            )
    out = pd.DataFrame(rows)
    out["spearman_p_fdr_bh"] = bh_fdr(out["spearman_p"].tolist())
    return out.sort_values(["abs_spearman_rho", "abs_pearson_r"], ascending=[False, False])


def merged_mean_cross_task(t5: pd.DataFrame, t6: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    t5_mean, t5_mean_cols = collapsed_t5(t5)
    merged = t5_mean.merge(t6[["subject_id"] + T6_COLS], on="subject_id", how="inner")
    return merged, t5_mean_cols


def merged_direct_cross_task(t5: pd.DataFrame, t6: pd.DataFrame, t5_cols: list[str]) -> pd.DataFrame:
    return t5[["subject_id"] + t5_cols].merge(t6[["subject_id"] + T6_COLS], on="subject_id", how="inner")


def spearman_value(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx, ddof=1) == 0 or np.std(ry, ddof=1) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman_ci(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    family: str,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for pair_idx, row in pairs.reset_index(drop=True).iterrows():
        a = str(row["parameter_a"])
        b = str(row["parameter_b"])
        valid = df[[a, b]].dropna()
        x = valid[a].to_numpy(dtype=float)
        y = valid[b].to_numpy(dtype=float)
        n = len(valid)
        rx = rankdata(x)
        ry = rankdata(y)
        idx = rng.integers(0, n, size=(n_boot, n))
        bx = rx[idx]
        by = ry[idx]
        bx = bx - bx.mean(axis=1, keepdims=True)
        by = by - by.mean(axis=1, keepdims=True)
        denom = np.sqrt((bx * bx).sum(axis=1) * (by * by).sum(axis=1))
        boot = np.divide((bx * by).sum(axis=1), denom, out=np.full(n_boot, np.nan), where=denom > 0)
        boot = boot[np.isfinite(boot)]
        observed = float(row["spearman_rho"]) if "spearman_rho" in row else spearman_value(x, y)
        rows.append(
            {
                "family": family,
                "parameter_a": a,
                "parameter_b": b,
                "n": int(n),
                "spearman_rho": observed,
                "rho_boot_mean": float(np.mean(boot)) if len(boot) else np.nan,
                "rho_ci_low": float(np.quantile(boot, 0.025)) if len(boot) else np.nan,
                "rho_ci_high": float(np.quantile(boot, 0.975)) if len(boot) else np.nan,
                "n_boot": int(n_boot),
                "n_boot_finite": int(len(boot)),
            }
        )
    return pd.DataFrame(rows)


def winsorized_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_low, x_high = np.quantile(x, [WINSOR_Q_LOW, WINSOR_Q_HIGH])
    y_low, y_high = np.quantile(y, [WINSOR_Q_LOW, WINSOR_Q_HIGH])
    return spearman_value(np.clip(x, x_low, x_high), np.clip(y, y_low, y_high))


def leave_one_out_max_delta(x: np.ndarray, y: np.ndarray, baseline: float) -> float:
    deltas: list[float] = []
    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        rho = spearman_value(x[keep], y[keep])
        if np.isfinite(rho):
            deltas.append(abs(rho - baseline))
    return float(max(deltas)) if deltas else np.nan


def outlier_sensitivity(df: pd.DataFrame, pairs: pd.DataFrame, family: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in pairs.iterrows():
        a = str(row["parameter_a"])
        b = str(row["parameter_b"])
        valid = df[[a, b]].dropna()
        x = valid[a].to_numpy(dtype=float)
        y = valid[b].to_numpy(dtype=float)
        rho = float(row["spearman_rho"]) if "spearman_rho" in row else spearman_value(x, y)
        rho_w = winsorized_spearman(x, y)
        loo_delta = leave_one_out_max_delta(x, y, rho)
        rows.append(
            {
                "family": family,
                "parameter_a": a,
                "parameter_b": b,
                "n": int(len(valid)),
                "spearman_rho": rho,
                "winsorized_01_99_spearman_rho": rho_w,
                "winsorized_delta": float(rho_w - rho) if np.isfinite(rho_w) and np.isfinite(rho) else np.nan,
                "leave_one_out_max_abs_delta": loo_delta,
            }
        )
    return pd.DataFrame(rows)


def add_stability_label(sensitivity: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    out = sensitivity.merge(
        boot[["family", "parameter_a", "parameter_b", "rho_ci_low", "rho_ci_high"]],
        on=["family", "parameter_a", "parameter_b"],
        how="left",
    )
    labels: list[str] = []
    reasons: list[str] = []
    for _, row in out.iterrows():
        rho = float(row["spearman_rho"])
        win = float(row["winsorized_01_99_spearman_rho"])
        ci_low = float(row["rho_ci_low"])
        ci_high = float(row["rho_ci_high"])
        loo = float(row["leave_one_out_max_abs_delta"])
        ci_same_side = np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_low > 0 or ci_high < 0)
        win_same_direction = np.isfinite(win) and (rho == 0 or np.sign(win) == np.sign(rho))
        loo_small = np.isfinite(loo) and loo <= LOO_STABLE_DELTA
        if ci_same_side and win_same_direction and loo_small:
            labels.append("stable")
            reasons.append("CI excludes zero; winsorized direction unchanged; leave-one-out shift small")
        else:
            labels.append("needs_caution")
            failed = []
            if not ci_same_side:
                failed.append("CI crosses zero")
            if not win_same_direction:
                failed.append("winsorized direction changes")
            if not loo_small:
                failed.append("leave-one-out shift > 0.10")
            reasons.append("; ".join(failed) if failed else "stability rule not fully satisfied")
    out["stability_label"] = labels
    out["stability_reason"] = reasons
    out["abs_spearman_rho"] = out["spearman_rho"].abs()
    return out.sort_values(["abs_spearman_rho", "leave_one_out_max_abs_delta"], ascending=[False, True])


def modeling_recommendations(vif: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    direct_stable = sensitivity[
        (sensitivity["family"] == "T5_T6_direct_cross_task") & (sensitivity["stability_label"] == "stable")
    ].head(8)
    stable_pairs = "; ".join(f"{r.parameter_a} vs {r.parameter_b}" for r in direct_stable.itertuples())
    return pd.DataFrame(
        [
            {
                "parameter_set": "T5 condition parameters",
                "status": "usable_with_caution",
                "evidence": "T5 matrix is full rank; several T_half/H_bar_norm columns show moderate VIF.",
                "recommendation": "Use direct parameters for exploratory maps; for regression use selected conditions, parameter-family means, PCA, or regularization.",
            },
            {
                "parameter_set": "T6 pm",
                "status": "requires_transformation",
                "evidence": "pm_positive + pm_negative + pm_neutral = 1, producing exact rank deficit and infinite VIF.",
                "recommendation": "Do not enter all three pm columns in ordinary regression; use two components or compositional/log-ratio transformation.",
            },
            {
                "parameter_set": "T6 Tmax",
                "status": "collinear_as_family",
                "evidence": "Tmax columns show high VIF when modeled together with the full T6 set.",
                "recommendation": "Avoid unselected simultaneous use; consider contrasts, one representative dimension, PCA, or regularization.",
            },
            {
                "parameter_set": "T6 dc",
                "status": "relatively_independent",
                "evidence": "dc VIF is low in the current T6 parameter set.",
                "recommendation": "Can be retained as a separate candidate dimension, with exploratory interpretation boundaries.",
            },
            {
                "parameter_set": "T5-T6 cross-task candidates",
                "status": "exploratory_candidates",
                "evidence": stable_pairs if stable_pairs else "No top direct cross-task relation satisfied the full stability rule.",
                "recommendation": "Use stable direct-parameter pairs as prioritized candidates for later scale or mechanism-facing models.",
            },
        ]
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(corr: pd.DataFrame, cols: list[str], title: str, stem: str, method_col: str = "spearman_rho") -> None:
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    for _, row in corr.iterrows():
        a = row["parameter_a"]
        b = row["parameter_b"]
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = float(row[method_col])
            mat.loc[b, a] = float(row[method_col])
    fig, ax = plt.subplots(figsize=(max(5.8, len(cols) * 0.46), max(4.8, len(cols) * 0.36)), constrained_layout=True)
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = float(mat.iloc[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(val) > 0.55 else "#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman rho")
    ax.set_title(title, fontsize=11, fontweight="bold")
    save_figure(fig, stem)


def plot_rect_heatmap(corr: pd.DataFrame, row_cols: list[str], col_cols: list[str], title: str, stem: str) -> None:
    mat = corr.pivot(index="parameter_a", columns="parameter_b", values="spearman_rho").reindex(index=row_cols, columns=col_cols)
    fig, ax = plt.subplots(figsize=(max(6.4, len(col_cols) * 0.72), max(5.2, len(row_cols) * 0.32)), constrained_layout=True)
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(col_cols)))
    ax.set_yticks(np.arange(len(row_cols)))
    ax.set_xticklabels(col_cols, rotation=42, ha="right")
    ax.set_yticklabels(row_cols)
    for i in range(len(row_cols)):
        for j in range(len(col_cols)):
            val = float(mat.iloc[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(val) > 0.55 else "#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman rho")
    ax.set_title(title, fontsize=11, fontweight="bold")
    save_figure(fig, stem)


def plot_vif(vif: pd.DataFrame, stem: str, title: str) -> None:
    plot = vif.copy()
    plot["VIF_plot"] = plot["VIF"].replace(np.inf, np.nan)
    finite_max = plot["VIF_plot"].max(skipna=True)
    cap = max(10.0, finite_max if np.isfinite(finite_max) else 10.0)
    plot["VIF_plot"] = plot["VIF_plot"].fillna(cap * 1.15)
    fig, ax = plt.subplots(figsize=(7.2, max(3.4, len(plot) * 0.25)), constrained_layout=True)
    y = np.arange(len(plot))
    colors = ["#D55E00" if not np.isfinite(v) or v >= 10 else "#E69F00" if v >= 5 else "#0072B2" for v in plot["VIF"]]
    bars = ax.barh(y, plot["VIF_plot"], color=colors, alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["parameter"])
    ax.invert_yaxis()
    ax.axvline(5, color="#666666", linestyle="--", linewidth=0.9)
    ax.axvline(10, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_xlabel("VIF (infinite values shown capped)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for bar, vif_value in zip(bars, plot["VIF"]):
        label = "inf" if not np.isfinite(vif_value) else f"{vif_value:.1f}"
        ax.text(bar.get_width() + cap * 0.015, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=7)
    save_figure(fig, stem)


def plot_bootstrap_ci_top(boot_direct: pd.DataFrame) -> None:
    plot = boot_direct.reindex(boot_direct["spearman_rho"].abs().sort_values(ascending=False).index).head(12).copy()
    plot = plot.iloc[::-1].reset_index(drop=True)
    labels = [f"{a} vs {b}" for a, b in zip(plot["parameter_a"], plot["parameter_b"])]
    y = np.arange(len(plot))
    x = plot["spearman_rho"].to_numpy(dtype=float)
    xerr = np.vstack(
        [
            x - plot["rho_ci_low"].to_numpy(dtype=float),
            plot["rho_ci_high"].to_numpy(dtype=float) - x,
        ]
    )
    colors = ["#0072B2" if val >= 0 else "#D55E00" for val in x]
    fig, ax = plt.subplots(figsize=(7.6, max(3.8, len(plot) * 0.34)), constrained_layout=True)
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#555555", elinewidth=1.0, capsize=3, zorder=1)
    ax.scatter(x, y, s=34, color=colors, edgecolor="white", linewidth=0.4, zorder=2)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman rho with bootstrap 95% CI")
    ax.set_title("Figure 6. Bootstrap CI for top direct T5-T6 correlations", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    save_figure(fig, "Figure_06_T5_T6_direct_bootstrap_ci_top_correlations")


def plot_outlier_sensitivity_top(sensitivity: pd.DataFrame) -> None:
    families = ["T5_T6_direct_cross_task", "T5_T6_cross_task"]
    plot = sensitivity[sensitivity["family"].isin(families)].copy()
    plot = plot.reindex(plot["spearman_rho"].abs().sort_values(ascending=False).index).head(12)
    plot = plot.iloc[::-1].reset_index(drop=True)
    labels = [f"{a} vs {b}" for a, b in zip(plot["parameter_a"], plot["parameter_b"])]
    y = np.arange(len(plot))
    orig = plot["spearman_rho"].to_numpy(dtype=float)
    win = plot["winsorized_01_99_spearman_rho"].to_numpy(dtype=float)
    loo = plot["leave_one_out_max_abs_delta"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, max(3.8, len(plot) * 0.34)), constrained_layout=True)
    for yi, x0, x1 in zip(y, orig, win):
        ax.plot([x0, x1], [yi, yi], color="#888888", linewidth=1.0, zorder=1)
    ax.scatter(orig, y, s=34, color="#0072B2", label="Original rho", zorder=2)
    ax.scatter(win, y, s=34, color="#E69F00", marker="D", label="Winsorized rho", zorder=3)
    for yi, shift in zip(y, loo):
        ax.text(1.01, yi, f"LOO {shift:.2f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6, clip_on=False)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Figure 7. Outlier sensitivity for top cross-task correlations", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    ax.legend(loc="lower right")
    save_figure(fig, "Figure_07_outlier_sensitivity_top_correlations")


def fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "inf" if x > 0 else "NA"
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
            if col in {"spearman_p", "spearman_p_fdr_bh"}:
                out[col] = out[col].map(lambda value: fmt_p(float(value)))
            else:
                out[col] = out[col].map(lambda value: fmt(float(value), 3))
    headers = list(out.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_report(
    t5: pd.DataFrame,
    t6: pd.DataFrame,
    t5_cols: list[str],
    t6_cols: list[str],
    t5_corr: pd.DataFrame,
    t6_corr: pd.DataFrame,
    cross_corr: pd.DataFrame,
    direct_cross_corr: pd.DataFrame,
    diagnostics: pd.DataFrame,
    vif: pd.DataFrame,
    boot_all: pd.DataFrame,
    sensitivity: pd.DataFrame,
    modeling_recommendations: pd.DataFrame,
) -> None:
    pm_sum_max = float((t6[["pm_positive", "pm_negative", "pm_neutral"]].sum(axis=1) - 1).abs().max())
    t5_high = t5_corr[(t5_corr["abs_spearman_rho"] >= 0.70) | (t5_corr["abs_pearson_r"] >= 0.80)]
    t6_high = t6_corr[(t6_corr["abs_spearman_rho"] >= 0.70) | (t6_corr["abs_pearson_r"] >= 0.80)]
    cross_top = cross_corr.head(10)
    direct_cross_top = direct_cross_corr.head(15)
    boot_direct_top = boot_all[boot_all["family"] == "T5_T6_direct_cross_task"].copy()
    boot_direct_top = boot_direct_top.reindex(boot_direct_top["spearman_rho"].abs().sort_values(ascending=False).index).head(12)
    sensitivity_top = sensitivity[sensitivity["family"].isin(["T5_T6_direct_cross_task", "T5_T6_cross_task"])].head(12)
    stable_top = sensitivity_top[sensitivity_top["stability_label"] == "stable"]
    lines = [
        "# N118 参数间关系与共线性分析报告",
        "",
        "日期：2026-07-29",
        "",
        "## 1. 数据与参数口径",
        "",
        "本报告基于用户修改后的 118 人参数数据，专门分析参数之间的关系、共线性风险和稳健性。分析单位是被试；T5 与 T6 均为 N=118，按 `subject_id` 合并后仍为 N=118。",
        "",
        f"- 数据目录：`{DATA}`",
        f"- T5 当前参数列数：{len(t5_cols)}，实际列为 `{', '.join(t5_cols)}`。",
        f"- T6 当前参数列数：{len(t6_cols)}，实际列为 `{', '.join(t6_cols)}`。",
        "",
        "本报告只使用当前 `t5_results.csv` 和 `t6_results.csv`，不使用旧 fit-record 参数值。当前 T5 数据只包含 `T_half/H_bar_norm/I_focus`，因此本文不再沿用旧导出参数口径。",
        "",
        "## 2. 参数自身是否可进入分析",
        "",
        "这一步回答的问题是：参数表本身是否完整、是否存在精确线性依赖，以及哪些参数不能未经处理就一起进入普通回归模型。",
        "",
        markdown_table(diagnostics),
        "",
        "VIF 解释：VIF > 5 提示需要谨慎，VIF > 10 通常表示严重共线性；`inf` 表示存在精确线性依赖。",
        "",
        "T5 VIF 最高项：",
        "",
        markdown_table(vif[vif["family"] == "T5_full"].head(12)),
        "",
        "T6 VIF：",
        "",
        markdown_table(vif[vif["family"] == "T6_full"].head(12)),
        "",
        "![T5 VIF](figures/Figure_03_T5_VIF.png)",
        "",
        "![T6 VIF](figures/Figure_04_T6_VIF.png)",
        "",
        f"T6 的 `pm` 三列满足组成约束，`pm_positive + pm_negative + pm_neutral = 1`，当前最大数值误差为 {pm_sum_max:.3e}。因此三个 `pm` 不能同时作为普通回归模型的自变量。",
        "",
        "## 3. 任务内参数结构",
        "",
        "这一层分别考察 T5 和 T6 内部参数是否形成稳定结构。它回答的是：同一个任务内部的参数是互补、冗余，还是存在共线性风险。",
        "",
        "### 3.1 T5 参数内部关系",
        "",
        "![T5 heatmap](figures/Figure_01_T5_within_parameter_spearman_heatmap.png)",
        "",
        "图 1 回答的问题是：当前 T5 12 个参数列之间是否存在强单调关系。颜色和数字为 Spearman rho。",
        "",
        "T5 高相关组合如下：",
        "",
        markdown_table(t5_high[["parameter_a", "parameter_b", "n", "pearson_r", "spearman_rho", "spearman_p_fdr_bh"]].head(20)) if len(t5_high) else "当前阈值下未发现高相关组合。",
        "",
        "从当前新数据看，T5 只包含 `T_half/H_bar_norm/I_focus` 三个参数族。当前 T5 参数间主要需要关注的是同类参数跨图片条件的相关，以及 `H_bar_norm` 与 `I_focus` 等参数族之间是否形成可替代信息。",
        "",
        "### 3.2 T6 参数内部关系",
        "",
        "![T6 heatmap](figures/Figure_02_T6_within_parameter_spearman_heatmap.png)",
        "",
        "图 2 回答的问题是：当前 T6 的 `pm/Tmax/dc` 参数之间是否存在强关系或组成约束。",
        "",
        "T6 高相关组合如下：",
        "",
        markdown_table(t6_high[["parameter_a", "parameter_b", "n", "pearson_r", "spearman_rho", "spearman_p_fdr_bh"]].head(20)) if len(t6_high) else "当前阈值下未发现 |Spearman rho| >= 0.70 或 |Pearson r| >= 0.80 的非平凡高相关组合；但 `pm` 三列存在精确组成约束。",
        "",
        "`dc` 的 VIF 较低，说明它在当前 T6 参数集中相对独立。`pm` 的共线性主要来自三分量和为 1，而不是来自某个单独 pair 的强 Spearman 相关。",
        "",
        "## 4. 跨任务参数关系",
        "",
        "跨任务相关分成两层：第一层直接使用 T5 每个条件参数与 T6 参数做相关；第二层把 T5 四类图片先按被试取均值后，再与 T6 参数做摘要相关。直接参数相关用于保留条件信息，均值相关用于看整体参数族趋势。",
        "",
        "### 4.1 直接参数相关：T5 12 个条件参数 x T6 7 个参数",
        "",
        "![Direct T5-T6 heatmap](figures/Figure_05_T5_T6_direct_parameter_spearman_heatmap.png)",
        "",
        "图 5 回答的问题是：不折叠 T5 图片条件时，哪些具体 T5 条件参数与 T6 参数关系最强。颜色和数字为 Spearman rho。",
        "",
        "直接参数相关中绝对相关最高的组合如下：",
        "",
        markdown_table(direct_cross_top[["parameter_a", "parameter_b", "n", "pearson_r", "spearman_rho", "spearman_p", "spearman_p_fdr_bh"]]),
        "",
        "直接相关表比均值表更适合回答“哪个图片条件下的 T5 参数与哪个 T6 参数有关”。这些结果仍是探索性相关，不代表量表、诊断或因果结论。",
        "",
        "### 4.2 T5 参数族均值相关：摘要层结果",
        "",
        "T5 四类图片先按被试取均值，再与 T6 参数做探索性 Spearman 相关。绝对相关最高的组合如下：",
        "",
        markdown_table(cross_top[["parameter_a", "parameter_b", "n", "pearson_r", "spearman_rho", "spearman_p", "spearman_p_fdr_bh"]]),
        "",
        "均值相关可以减少条件数量，但会丢掉具体图片条件信息。因此后续如果要解释参数机制，应优先查看直接参数相关，再用均值相关作为整体趋势补充。",
        "",
        "## 5. 稳健性分析",
        "",
        f"这一步回答的问题是：前面看到的相关关系是否依赖少数极端值，相关方向和大小在重采样下是否稳定。这里使用 {N_BOOT} 次 bootstrap Spearman 相关置信区间、1%/99% winsorized 敏感性，以及 leave-one-out 最大偏移。",
        "",
        "![Bootstrap CI](figures/Figure_06_T5_T6_direct_bootstrap_ci_top_correlations.png)",
        "",
        "图 6 显示直接跨任务相关中绝对相关较高组合的 bootstrap 95% CI。CI 不跨 0 的组合更适合作为后续建模候选，但仍只能解释为探索性参数关系。",
        "",
        markdown_table(boot_direct_top[["parameter_a", "parameter_b", "n", "spearman_rho", "rho_boot_mean", "rho_ci_low", "rho_ci_high", "n_boot"]]),
        "",
        "![Outlier sensitivity](figures/Figure_07_outlier_sensitivity_top_correlations.png)",
        "",
        "图 7 比较原始 Spearman rho 与 winsorized 后的 rho，并标注 leave-one-out 最大偏移。若 winsorized 后方向保持一致、bootstrap CI 不跨 0，且 leave-one-out 最大偏移较小，则该关系被标记为较稳定。",
        "",
        markdown_table(sensitivity_top[["family", "parameter_a", "parameter_b", "spearman_rho", "winsorized_01_99_spearman_rho", "leave_one_out_max_abs_delta", "rho_ci_low", "rho_ci_high", "stability_label"]]),
        "",
        f"在当前前 12 个跨任务候选关系中，稳定性标签为 `stable` 的组合数为 {len(stable_top)}。这个结果用于筛选后续模型候选，不应单独写成机制证明。",
        "",
        "## 6. 论文解释边界",
        "",
        "- 本报告只分析参数表本身，不纳入量表、诊断分组、高低分组或纵向变化。",
        "- 所有相关都是被试层面的探索性参数关系，不能写成因果关系、临床诊断标志或神经机制证明。",
        "- T6 的 `pm` 是组成数据，普通相关和 VIF 可用于识别结构约束，但后续回归需要只放两个分量或使用 compositional/log-ratio 变换。",
        "- bootstrap CI 和极端值敏感性用于判断结果是否稳定，不把探索性相关升级为确认性结论。",
        "",
        "## 7. 后续建模建议",
        "",
        "- T5：当前新数据建模时应只依据实际存在的 `T_half/H_bar_norm/I_focus` 列。若同一参数族跨图片条件高度相关，建议用条件差值、均值、PCA 或正则化模型，而不是无筛选地全部放入普通回归。",
        "- T6：三个 `pm` 不能同时进入普通回归；建议只放两个 `pm`，或使用 compositional/log-ratio 变换。",
        "- T6：`dc` 当前相对独立，可作为单独维度保留。",
        "- 跨任务：应同时保留直接参数相关和 T5 参数族均值相关；直接相关用于定位具体条件，均值相关用于摘要整体趋势。",
        "- 跨任务：相关最高的 T5-T6 组合可作为下一步量表关联或机制解释的候选，但需要单独建模与复核。",
        "",
        "参数使用建议表：",
        "",
        markdown_table(modeling_recommendations),
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_plotting()
    ensure_dirs()
    t5, t6, t5_cols, t6_cols = load_data()
    t5_corr = pairwise_correlations(t5, t5_cols, "T5_within")
    t6_corr = pairwise_correlations(t6, t6_cols, "T6_within")
    cross_corr = cross_task_correlations(t5, t6)
    direct_cross_corr = direct_cross_task_correlations(t5, t6, t5_cols)
    mean_merged, mean_cols = merged_mean_cross_task(t5, t6)
    direct_merged = merged_direct_cross_task(t5, t6, t5_cols)
    diagnostics = pd.DataFrame(
        [
            matrix_diagnostics(t5, t5_cols, "T5_full"),
            matrix_diagnostics(t6, t6_cols, "T6_full"),
        ]
    )
    vif = pd.concat([vif_table(t5, t5_cols, "T5_full"), vif_table(t6, t6_cols, "T6_full")], ignore_index=True)

    boot_t5 = bootstrap_spearman_ci(t5, t5_corr, "T5_within", seed=BOOT_SEED + 1)
    boot_t6 = bootstrap_spearman_ci(t6, t6_corr, "T6_within", seed=BOOT_SEED + 2)
    boot_direct = bootstrap_spearman_ci(direct_merged, direct_cross_corr, "T5_T6_direct_cross_task", seed=BOOT_SEED + 3)
    boot_cross = bootstrap_spearman_ci(mean_merged, cross_corr, "T5_T6_cross_task", seed=BOOT_SEED + 4)
    boot_all = pd.concat([boot_t5, boot_t6, boot_direct, boot_cross], ignore_index=True)

    sens_t5 = outlier_sensitivity(t5, t5_corr, "T5_within")
    sens_t6 = outlier_sensitivity(t6, t6_corr, "T6_within")
    sens_direct = outlier_sensitivity(direct_merged, direct_cross_corr, "T5_T6_direct_cross_task")
    sens_cross = outlier_sensitivity(mean_merged, cross_corr, "T5_T6_cross_task")
    sensitivity = add_stability_label(pd.concat([sens_t5, sens_t6, sens_direct, sens_cross], ignore_index=True), boot_all)
    recs = modeling_recommendations(vif, sensitivity)

    t5_corr.to_csv(TAB_DIR / "t5_within_parameter_correlations_N118.csv", index=False, encoding="utf-8-sig")
    t6_corr.to_csv(TAB_DIR / "t6_within_parameter_correlations_N118.csv", index=False, encoding="utf-8-sig")
    cross_corr.to_csv(TAB_DIR / "t5_t6_cross_task_parameter_correlations_N118.csv", index=False, encoding="utf-8-sig")
    direct_cross_corr.to_csv(TAB_DIR / "t5_t6_direct_parameter_correlations_N118.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(TAB_DIR / "parameter_matrix_rank_diagnostics_N118.csv", index=False, encoding="utf-8-sig")
    vif.to_csv(TAB_DIR / "parameter_vif_diagnostics_N118.csv", index=False, encoding="utf-8-sig")
    boot_t5.to_csv(TAB_DIR / "t5_within_parameter_correlations_bootstrap_N118.csv", index=False, encoding="utf-8-sig")
    boot_t6.to_csv(TAB_DIR / "t6_within_parameter_correlations_bootstrap_N118.csv", index=False, encoding="utf-8-sig")
    boot_direct.to_csv(TAB_DIR / "t5_t6_direct_parameter_correlations_bootstrap_N118.csv", index=False, encoding="utf-8-sig")
    boot_cross.to_csv(TAB_DIR / "t5_t6_cross_task_parameter_correlations_bootstrap_N118.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(TAB_DIR / "parameter_correlation_outlier_sensitivity_N118.csv", index=False, encoding="utf-8-sig")
    recs.to_csv(TAB_DIR / "parameter_modeling_recommendations_N118.csv", index=False, encoding="utf-8-sig")

    plot_heatmap(t5_corr, t5_cols, "Figure 1. T5 within-parameter Spearman correlations", "Figure_01_T5_within_parameter_spearman_heatmap")
    plot_heatmap(t6_corr, t6_cols, "Figure 2. T6 within-parameter Spearman correlations", "Figure_02_T6_within_parameter_spearman_heatmap")
    plot_vif(vif[vif["family"] == "T5_full"], "Figure_03_T5_VIF", "Figure 3. T5 multicollinearity diagnostics")
    plot_vif(vif[vif["family"] == "T6_full"], "Figure_04_T6_VIF", "Figure 4. T6 multicollinearity diagnostics")
    plot_rect_heatmap(direct_cross_corr, t5_cols, t6_cols, "Figure 5. Direct T5-T6 parameter Spearman correlations", "Figure_05_T5_T6_direct_parameter_spearman_heatmap")
    plot_bootstrap_ci_top(boot_direct)
    plot_outlier_sensitivity_top(sensitivity)
    write_report(t5, t6, t5_cols, t6_cols, t5_corr, t6_corr, cross_corr, direct_cross_corr, diagnostics, vif, boot_all, sensitivity, recs)
    print(f"Generated parameter relationship bundle: {BASE}")


if __name__ == "__main__":
    main()
