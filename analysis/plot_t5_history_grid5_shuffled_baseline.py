#!/usr/bin/env python3
"""Task-level T5 history-persistence evidence using a 5x5 shuffled baseline.

This script answers one narrow question: does the T5 gaze trajectory contain
history dependence at the task level? It folds emotions, avoids 10x10
comparisons, and compares observed 5x5 lagged same-cell similarity against a
within-trial shuffled spatial baseline.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-t5-grid5-baseline")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/mpl-t5-grid5-baseline")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rebuild_n80_direct_model_free_from_raw import (  # noqa: E402
    TIMESTAMP_UNITS_PER_MS,
    read_csv,
    row_timestamp,
    screen_valid_point,
    t5_rect,
)
from analyze_t5_three_process_model_free import (  # noqa: E402
    LAGS_MS,
    discover_workspace_t5_pairs,
    get_record,
    iter_eye_rows,
    mean_or_nan,
    row_trigger,
    sd_or_nan,
)

GRID_SIZE = 5
N_SHUFFLES = 0
RNG_SEED = 20260628

TRIAL_CSV = SCRIPT_DIR / "raw重算_direct_model_free_N80_with_samsung" / "T5_raw_direct_trial_N80.csv"
OUT_DIR = SCRIPT_DIR / "core_model_free_t5_history_grid5_task_evidence_N80"
FIG_DIR = OUT_DIR / "figures"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grid_cell(nx: float, ny: float) -> tuple[int, int]:
    return min(int(nx * GRID_SIZE), GRID_SIZE - 1), min(int(ny * GRID_SIZE), GRID_SIZE - 1)


def lagged_same_cell_similarity(timestamps: list[int], cells: list[tuple[int, int]], lag_ms: int) -> tuple[float, int]:
    if len(timestamps) < 2:
        return math.nan, 0
    lag_units = lag_ms * TIMESTAMP_UNITS_PER_MS
    matches = 0
    usable = 0
    for idx, ts in enumerate(timestamps):
        target = ts - lag_units
        prev_idx = bisect_right(timestamps, target) - 1
        if prev_idx < 0:
            continue
        usable += 1
        matches += int(cells[idx] == cells[prev_idx])
    return (matches / usable if usable else math.nan), usable


def history_empty(row: dict[str, str], status: str) -> dict[str, object]:
    out: dict[str, object] = {
        "subject": row.get("subject", ""),
        "subject_id": row.get("subject_id", ""),
        "trial": row.get("trial", ""),
        "emotion": row.get("emotion", ""),
        "cut_status": status,
        "image_valid_points": 0,
        "grid_size": GRID_SIZE,
        "n_shuffles": N_SHUFFLES,
        "observed_mean": math.nan,
        "shuffled_mean": math.nan,
        "excess_mean": math.nan,
    }
    for lag in LAGS_MS:
        out[f"observed_{lag}ms"] = math.nan
        out[f"shuffled_{lag}ms"] = math.nan
        out[f"excess_{lag}ms"] = math.nan
        out[f"n_{lag}ms"] = 0
    return out


def resolve_paths(row: dict[str, str], workspace_pairs: dict[str, tuple[Path, str]]) -> tuple[Path, str]:
    records_path = Path(row["raw_records_path"])
    eyetrack_label = row["raw_eyetrack_path"]
    eyetrack_ok = all(Path(part).exists() for part in eyetrack_label.split(";") if part)
    if (not records_path.exists() or not eyetrack_ok) and row.get("subject_id") in workspace_pairs:
        records_path, eyetrack_label = workspace_pairs[row["subject_id"]]
    return records_path, eyetrack_label


def make_trial_specs(rows: list[dict[str, str]], records_cache: dict[Path, dict[int, dict[str, str]]], workspace_pairs: dict[str, tuple[Path, str]]):
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    failed: list[dict[str, object]] = []
    for row in rows:
        if row.get("task_type") != "T5":
            continue
        try:
            records_path, eyetrack_label = resolve_paths(row, workspace_pairs)
            trial = int(float(row["trial"]))
            record = get_record(records_cache, records_path, eyetrack_label, trial)
            if record is None:
                failed.append(history_empty(row, "missing_record"))
                continue
            spec = {
                "row": row,
                "trial": trial,
                "start_trigger": int(float(row["start_trigger"])),
                "end_trigger": int(float(row["end_trigger"])),
                "rect": t5_rect(record),
                "timestamps": [],
                "cells": [],
            }
            grouped[eyetrack_label].append(spec)
        except (KeyError, ValueError, SyntaxError):
            failed.append(history_empty(row, "parse_error"))
    for specs in grouped.values():
        specs.sort(key=lambda spec: int(spec["start_trigger"]))
    return grouped, failed


def add_gaze_to_spec(spec: dict[str, object], gaze_row: dict[str, str]) -> None:
    ts = row_timestamp(gaze_row)
    point = screen_valid_point(gaze_row)
    rect = spec["rect"]
    if ts is None or point is None or not rect.contains(*point):
        return
    nx = min(max((point[0] - rect.x) / rect.w, 0.0), 1.0)
    ny = min(max((point[1] - rect.y) / rect.h, 0.0), 1.0)
    spec["timestamps"].append(ts)
    spec["cells"].append(grid_cell(nx, ny))


def shuffled_baseline_value(cells: list[tuple[int, int]]) -> float:
    """Expected same-cell probability after shuffling cell labels within trial.

    This preserves the trial's spatial occupancy distribution while removing
    temporal order. The finite-sample without-replacement expectation is used,
    so repeated draws cannot reuse the same gaze sample.
    """
    n = len(cells)
    if n < 2:
        return math.nan
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for cell in cells:
        counts[cell] += 1
    same_pairs = sum(count * (count - 1) for count in counts.values())
    return same_pairs / (n * (n - 1))


def finalize_spec(spec: dict[str, object], status: str = "cut") -> dict[str, object]:
    row = spec["row"]
    timestamps = spec["timestamps"]
    cells = spec["cells"]
    if len(cells) < 2:
        return history_empty(row, "no_image_points" if status == "cut" else status)

    observed = {}
    counts = {}
    for lag in LAGS_MS:
        value, count = lagged_same_cell_similarity(timestamps, cells, lag)
        observed[lag] = value
        counts[lag] = count
    baseline = shuffled_baseline_value(cells)
    shuffled = {lag: baseline for lag in LAGS_MS}
    excess = {lag: observed[lag] - shuffled[lag] for lag in LAGS_MS}

    out: dict[str, object] = {
        "subject": row["subject"],
        "subject_id": row["subject_id"],
        "trial": row["trial"],
        "emotion": row["emotion"],
        "cut_status": status,
        "image_valid_points": len(cells),
        "grid_size": GRID_SIZE,
        "n_shuffles": N_SHUFFLES,
        "observed_mean": mean_or_nan(list(observed.values())),
        "shuffled_mean": mean_or_nan(list(shuffled.values())),
        "excess_mean": mean_or_nan(list(excess.values())),
    }
    for lag in LAGS_MS:
        out[f"observed_{lag}ms"] = observed[lag]
        out[f"shuffled_{lag}ms"] = shuffled[lag]
        out[f"excess_{lag}ms"] = excess[lag]
        out[f"n_{lag}ms"] = counts[lag]
    return out


def process_streaming_trials(grouped_specs: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    total_groups = len(grouped_specs)
    for group_idx, (eyetrack_label, specs) in enumerate(sorted(grouped_specs.items()), start=1):
        start_map = {int(spec["start_trigger"]): spec for spec in specs}
        remaining = set(id(spec) for spec in specs)
        active: dict[str, object] | None = None
        for gaze_row in iter_eye_rows(eyetrack_label):
            trigger = row_trigger(gaze_row)
            if active is not None and trigger == int(active["end_trigger"]):
                output.append(finalize_spec(active))
                remaining.discard(id(active))
                active = None
                continue
            if active is not None:
                add_gaze_to_spec(active, gaze_row)
            if trigger in start_map:
                active = start_map[trigger]
                continue
        if active is not None:
            output.append(finalize_spec(active, "missing_end_trigger"))
            remaining.discard(id(active))
        for spec in specs:
            if id(spec) in remaining:
                output.append(history_empty(spec["row"], "missing_trigger_window"))
        if group_idx % 10 == 0 or group_idx == total_groups:
            print(f"processed_eyetrack_groups={group_idx}/{total_groups}", flush=True)
    return output


def ci95(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    delta = 1.96 * stdev(vals) / math.sqrt(len(vals))
    return mean(vals) - delta, mean(vals) + delta


def sign_flip_p_value(values: np.ndarray) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(vals)
    if n == 0:
        return math.nan
    observed = float(vals.mean())
    rng = np.random.default_rng(RNG_SEED + 17)
    reps = 200000
    batch = 20000
    hits = 0
    done = 0
    while done < reps:
        size = min(batch, reps - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, n))
        null_means = (signs * vals).mean(axis=1)
        hits += int(np.sum(null_means >= observed))
        done += size
    return (hits + 1) / (reps + 1)


def summarize_subjects(trial_rows: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(trial_rows)
    valid = df[df["cut_status"] == "cut"].copy()
    metric_cols = ["observed_mean", "shuffled_mean", "excess_mean"]
    for lag in LAGS_MS:
        metric_cols.extend([f"observed_{lag}ms", f"shuffled_{lag}ms", f"excess_{lag}ms"])
    subject = valid.groupby(["subject", "subject_id"], as_index=False).agg(
        valid_trial_count=("trial", "count"),
        image_valid_points=("image_valid_points", "sum"),
        **{col: (col, "mean") for col in metric_cols},
    )

    lag_rows = []
    for lag in LAGS_MS:
        obs = pd.to_numeric(subject[f"observed_{lag}ms"], errors="coerce")
        shuf = pd.to_numeric(subject[f"shuffled_{lag}ms"], errors="coerce")
        exc = pd.to_numeric(subject[f"excess_{lag}ms"], errors="coerce")
        obs_lo, obs_hi = ci95(obs.dropna().tolist())
        shuf_lo, shuf_hi = ci95(shuf.dropna().tolist())
        exc_lo, exc_hi = ci95(exc.dropna().tolist())
        lag_rows.append({
            "lag_ms": lag,
            "n_subjects": int(exc.notna().sum()),
            "observed_mean": float(obs.mean()),
            "observed_ci95_low": obs_lo,
            "observed_ci95_high": obs_hi,
            "shuffled_mean": float(shuf.mean()),
            "shuffled_ci95_low": shuf_lo,
            "shuffled_ci95_high": shuf_hi,
            "excess_mean": float(exc.mean()),
            "excess_ci95_low": exc_lo,
            "excess_ci95_high": exc_hi,
        })

    excess = pd.to_numeric(subject["excess_mean"], errors="coerce").dropna().to_numpy(dtype=float)
    lo, hi = ci95(excess.tolist())
    summary = pd.DataFrame([{
        "metric": "observed_minus_shuffled_mean",
        "n_subjects": len(excess),
        "mean": float(excess.mean()),
        "sd": float(excess.std(ddof=1)),
        "ci95_low": lo,
        "ci95_high": hi,
        "median": float(np.median(excess)),
        "min": float(excess.min()),
        "max": float(excess.max()),
        "sign_flip_one_sided_p": sign_flip_p_value(excess),
    }])
    return subject, pd.DataFrame(lag_rows), summary


def save_figures(subject: pd.DataFrame, lag_summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = lag_summary["lag_ms"].to_numpy(dtype=float)
    obs = lag_summary["observed_mean"].to_numpy(dtype=float)
    shuf = lag_summary["shuffled_mean"].to_numpy(dtype=float)
    obs_lo = lag_summary["observed_ci95_low"].to_numpy(dtype=float)
    obs_hi = lag_summary["observed_ci95_high"].to_numpy(dtype=float)
    shuf_lo = lag_summary["shuffled_ci95_low"].to_numpy(dtype=float)
    shuf_hi = lag_summary["shuffled_ci95_high"].to_numpy(dtype=float)
    ax.plot(x, obs, marker="o", lw=2.4, color="#0072B2", label="Observed trajectory")
    ax.fill_between(x, obs_lo, obs_hi, color="#0072B2", alpha=0.16, linewidth=0)
    ax.plot(x, shuf, marker="o", lw=2.4, color="#777777", label="Shuffled baseline")
    ax.fill_between(x, shuf_lo, shuf_hi, color="#777777", alpha=0.16, linewidth=0)
    ax.set_xticks(LAGS_MS)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Same-cell spatial similarity")
    ax.set_title("T5 history persistence against shuffled baseline (5x5 grid)")
    ax.grid(axis="both", color="#E5E5E5", linewidth=0.7)
    ax.legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Figure_T5_grid5_observed_vs_shuffled_lag_curve.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    values = pd.to_numeric(subject["excess_mean"], errors="coerce").dropna().to_numpy(dtype=float)
    lo, hi = ci95(values.tolist())
    avg = float(values.mean())
    fig, ax = plt.subplots(figsize=(4.8, 5.1))
    rng = np.random.default_rng(RNG_SEED + 29)
    jitter = rng.normal(0, 0.055, size=len(values))
    ax.scatter(np.zeros(len(values)) + jitter, values, s=24, color="#56B4E9", alpha=0.58, edgecolors="none")
    ax.errorbar([0], [avg], yerr=[[avg - lo], [hi - avg]], fmt="o", color="#222222", ecolor="#222222", capsize=5, lw=1.6, ms=7)
    ax.axhline(0, color="#333333", lw=1.1, linestyle="--")
    ax.set_xlim(-0.38, 0.38)
    ax.set_xticks([0])
    ax.set_xticklabels(["Observed - shuffled"])
    ax.set_ylabel("Excess same-cell similarity")
    ax.set_title("Task-level history effect (5x5 grid)")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Figure_T5_grid5_task_level_history_effect.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_summary(lag_summary: pd.DataFrame, effect_summary: pd.DataFrame) -> None:
    lines = [
        "# T5 5x5 Task-Level History Evidence",
        "",
        "Status: needs_review",
        "",
        "Question: Does the T5 task contain history dependence in gaze trajectories?",
        "",
        "Method: collapse emotions and compare observed 5x5 lagged same-cell spatial similarity with a within-trial shuffled spatial baseline.",
        "",
        "## Task-Level Effect",
        markdown_table(effect_summary),
        "",
        "## Lag-Level Observed vs Shuffled Summary",
        markdown_table(lag_summary),
        "",
        "## Figures",
        "- `figures/Figure_T5_grid5_observed_vs_shuffled_lag_curve.{png,pdf}`",
        "- `figures/Figure_T5_grid5_task_level_history_effect.{png,pdf}`",
    ]
    (OUT_DIR / "T5_grid5_task_level_history_evidence_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records_cache: dict[Path, dict[int, dict[str, str]]] = {}
    workspace_pairs = discover_workspace_t5_pairs()
    rows = read_csv(TRIAL_CSV)
    grouped, failed = make_trial_specs(rows, records_cache, workspace_pairs)
    trial_rows = [*failed, *process_streaming_trials(grouped)]
    trial_rows.sort(key=lambda row: (str(row.get("subject_id", "")), float(row.get("trial", 0) or 0)))
    subject, lag_summary, effect_summary = summarize_subjects(trial_rows)

    write_csv(OUT_DIR / "T5_grid5_task_level_history_trial_rows_N80.csv", trial_rows)
    subject.to_csv(OUT_DIR / "T5_grid5_task_level_history_subject_N80.csv", index=False, encoding="utf-8-sig")
    lag_summary.to_csv(OUT_DIR / "T5_grid5_observed_vs_shuffled_by_lag_N80.csv", index=False, encoding="utf-8-sig")
    effect_summary.to_csv(OUT_DIR / "T5_grid5_task_level_history_effect_summary_N80.csv", index=False, encoding="utf-8-sig")
    save_figures(subject, lag_summary)
    write_summary(lag_summary, effect_summary)

    print(f"trial_rows={len(trial_rows)}")
    print(f"subjects={subject['subject_id'].nunique()}")
    print(f"output_dir={OUT_DIR}")
    print(effect_summary.to_string(index=False))
    print(lag_summary.to_string(index=False))


if __name__ == "__main__":
    main()
