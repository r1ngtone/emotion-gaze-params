#!/usr/bin/env python3
"""Rebuild N=80 direct model-free evidence from raw gaze files.

This 0616 script intentionally does not read the 0606 direct/model-free CSVs.
It uses the N=80 subject frame plus raw `formal_records.csv` and
`formal_eyetrack.csv` pairs to recompute direct gaze indicators for the
attention processes used in the fifth task.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
EARLY_WINDOW_NS = 500_000_000
LAST_WINDOW_NS = 1_000_000_000
TIMESTAMP_UNITS_PER_MS = 1_000_000
GRID_SIZE = 10
GRID_CELL_COUNT = GRID_SIZE * GRID_SIZE


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


@dataclass(frozen=True)
class AOI:
    label: str
    emotion: str
    rect: Rect


@dataclass(frozen=True)
class RawPair:
    task_type: str
    subject: str
    subject_id: str
    records_path: Path
    eyetrack_paths: tuple[Path, ...]
    source_rank: int
    source_label: str

    @property
    def eyetrack_path_label(self) -> str:
        return ";".join(str(path) for path in self.eyetrack_paths)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_many_csv(paths: tuple[Path, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv(path))
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_pair(raw: str) -> tuple[float, float]:
    value = ast.literal_eval(raw)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"not a coordinate pair: {raw}")
    return float(value[0]), float(value[1])


def parse_rect(position: str, size: str) -> Rect:
    x, y = parse_pair(position)
    w, h = parse_pair(size)
    return Rect(x=x, y=y, w=w, h=h)


def normalize_emotion(raw: str) -> str:
    emotion = raw.strip().lower()
    return "positive" if emotion == "postive" else emotion


def trigger_indices(rows: list[dict[str, str]]) -> dict[int, int]:
    indices: dict[int, int] = {}
    for idx, row in enumerate(rows):
        raw = row.get("trigger", "").strip()
        if not raw:
            continue
        try:
            trigger = int(float(raw))
        except ValueError:
            continue
        if trigger > 0:
            indices[trigger] = idx
    return indices


def row_timestamp(row: dict[str, str]) -> int | None:
    try:
        return int(float(row["timestamp"]))
    except (KeyError, ValueError):
        return None


def screen_valid_point(row: dict[str, str]) -> tuple[float, float] | None:
    try:
        valid = int(float(row["bino_eye_valid"])) == 1
        x = float(row["bino_eye_gaze_position_x"])
        y = float(row["bino_eye_gaze_position_y"])
    except (KeyError, ValueError):
        return None
    if valid and 0 <= x <= SCREEN_WIDTH and 0 <= y <= SCREEN_HEIGHT:
        return x, y
    return None


def ratio(numerator: int | float, denominator: int | float) -> str:
    return f"{numerator / denominator:.6f}" if denominator else "0.000000"


def format_float(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def mean_float(rows: list[dict[str, object]], field: str) -> str:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "")
        if raw == "":
            continue
        values.append(float(raw))
    return f"{mean(values):.6f}" if values else ""


def sum_int(rows: list[dict[str, object]], field: str) -> int:
    return sum(int(float(row[field])) for row in rows)


def diff_string(left: str, right: str) -> str:
    if left == "" or right == "":
        return ""
    return f"{float(left) - float(right):.6f}"


def sd(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def value_range(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def entropy_norm(cells: list[tuple[int, int]]) -> float:
    if not cells:
        return 0.0
    counts = Counter(cells)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(GRID_CELL_COUNT)


def run_durations_ms(flags: list[bool], timestamps: list[int]) -> list[float]:
    durations: list[float] = []
    start_ts: int | None = None
    last_ts: int | None = None
    for flag, ts in zip(flags, timestamps):
        if flag and start_ts is None:
            start_ts = ts
        if flag:
            last_ts = ts
        if not flag and start_ts is not None and last_ts is not None:
            durations.append((last_ts - start_ts) / TIMESTAMP_UNITS_PER_MS)
            start_ts = None
            last_ts = None
    if start_ts is not None and last_ts is not None:
        durations.append((last_ts - start_ts) / TIMESTAMP_UNITS_PER_MS)
    return durations


def grid_cell(nx: float, ny: float) -> tuple[int, int]:
    return min(int(nx * GRID_SIZE), GRID_SIZE - 1), min(int(ny * GRID_SIZE), GRID_SIZE - 1)


def cell_center(cell: tuple[int, int]) -> tuple[float, float]:
    return (cell[0] + 0.5) / GRID_SIZE, (cell[1] + 0.5) / GRID_SIZE


def top_cell_ratio(counts: Counter[tuple[int, int]], top_cell_count: int, total: int) -> str:
    if total <= 0:
        return "0.000000"
    top_counts = sorted(counts.values(), reverse=True)[:top_cell_count]
    return ratio(sum(top_counts), total)


def hotspot_metrics(points: list[tuple[float, float]]) -> dict[str, object]:
    if not points:
        return {
            "hotspot_top_cell_ratio": "0.000000",
            "hotspot_top_5pct_cells_ratio": "0.000000",
            "hotspot_top_10pct_cells_ratio": "0.000000",
            "gaze_to_hotspot_mean_distance": "0.000000",
            "gaze_to_hotspot_median_distance": "0.000000",
        }
    cells = [grid_cell(nx, ny) for nx, ny in points]
    counts = Counter(cells)
    hotspot_cell, hotspot_count = counts.most_common(1)[0]
    hx, hy = cell_center(hotspot_cell)
    distances = [math.dist((nx, ny), (hx, hy)) for nx, ny in points]
    return {
        "hotspot_top_cell_ratio": ratio(hotspot_count, len(points)),
        "hotspot_top_5pct_cells_ratio": top_cell_ratio(counts, max(1, round(GRID_CELL_COUNT * 0.05)), len(points)),
        "hotspot_top_10pct_cells_ratio": top_cell_ratio(counts, max(1, round(GRID_CELL_COUNT * 0.10)), len(points)),
        "gaze_to_hotspot_mean_distance": f"{mean(distances):.6f}",
        "gaze_to_hotspot_median_distance": f"{median(distances):.6f}",
    }


def t6_aois(record: dict[str, str]) -> list[AOI]:
    return [
        AOI("left", normalize_emotion(record["左情绪类型"]), parse_rect(record["左图片坐标"], record["左图片大小"])),
        AOI("center", normalize_emotion(record["中情绪类型"]), parse_rect(record["中图片坐标"], record["中图片大小"])),
        AOI("right", normalize_emotion(record["右情绪类型"]), parse_rect(record["右图片坐标"], record["右图片大小"])),
    ]


def matched_aoi(point: tuple[float, float], aois: list[AOI]) -> AOI | None:
    x, y = point
    for aoi in aois:
        if aoi.rect.contains(x, y):
            return aoi
    return None


def t5_rect(record: dict[str, str]) -> Rect:
    return parse_rect(record["刺激图片坐标"], record["刺激图片显示尺寸"])


def empty_t6_metrics() -> dict[str, object]:
    return {
        "window_total_samples": 0,
        "screen_valid_points": 0,
        "aoi_points": 0,
        "negative_points": 0,
        "positive_points": 0,
        "neutral_points": 0,
        "negative_aoi_ratio": "0.000000",
        "positive_aoi_ratio": "0.000000",
        "negative_minus_positive_aoi_ratio": "0.000000",
        "positive_minus_negative_aoi_ratio": "0.000000",
        "first_negative_entry_ms": "",
        "first_positive_entry_ms": "",
        "negative_first_entry_advantage_ms": "",
        "positive_first_entry_delay_ms": "",
        "early_500ms_aoi_points": 0,
        "early_500ms_negative_points": 0,
        "early_500ms_positive_points": 0,
        "early_500ms_negative_ratio": "0.000000",
        "early_500ms_positive_ratio": "0.000000",
        "early_500ms_negative_minus_positive_ratio": "0.000000",
        "early_500ms_positive_minus_negative_ratio": "0.000000",
        "t6_left_aoi_ratio": "0.000000",
        "t6_center_aoi_ratio": "0.000000",
        "t6_right_aoi_ratio": "0.000000",
        "t6_aoi_coverage_count": 0,
        "t6_aoi_transition_count": 0,
        "t6_gaze_x_sd": "0.000000",
        "t6_gaze_x_range": "0.000000",
    }


def empty_t5_metrics() -> dict[str, object]:
    return {
        "window_total_samples": 0,
        "screen_valid_points": 0,
        "image_valid_points": 0,
        "valid_viewing_ratio": "0.000000",
        "late_half_viewing_ratio": "0.000000",
        "last_1000ms_viewing_ratio": "0.000000",
        "mean_continuous_viewing_run_ms": "",
        "leave_count": 0,
        "revisit_count": 0,
        "gaze_x_sd": "0.000000",
        "gaze_y_sd": "0.000000",
        "gaze_xy_dispersion": "0.000000",
        "normalized_coverage_width": "0.000000",
        "normalized_coverage_height": "0.000000",
        "normalized_coverage_area": "0.000000",
        "grid_coverage_ratio": "0.000000",
        "grid_entropy": "0.000000",
        "hotspot_top_cell_ratio": "0.000000",
        "hotspot_top_5pct_cells_ratio": "0.000000",
        "hotspot_top_10pct_cells_ratio": "0.000000",
        "gaze_to_hotspot_mean_distance": "0.000000",
        "gaze_to_hotspot_median_distance": "0.000000",
    }


def summarize_t6_trial(pair: RawPair, record: dict[str, str], eye_rows: list[dict[str, str]], indices: dict[int, int]) -> dict[str, object]:
    trial = int(float(record["试次轮次"]))
    start_trigger = trial * 100 + 1
    end_trigger = trial * 100 + 2
    start_idx = indices.get(start_trigger)
    end_idx = indices.get(end_trigger)
    base = {
        "task_type": "T6",
        "subject": pair.subject,
        "subject_id": pair.subject_id,
        "trial": trial,
        "emotion": "multi",
        "start_trigger": start_trigger,
        "end_trigger": end_trigger,
        "cut_status": "missing_trigger_window",
        "raw_records_path": str(pair.records_path),
        "raw_eyetrack_path": pair.eyetrack_path_label,
    }
    if start_idx is None or end_idx is None or start_idx >= end_idx:
        return {**base, **empty_t6_metrics()}
    start_ts = row_timestamp(eye_rows[start_idx])
    if start_ts is None:
        return {**base, **empty_t6_metrics()}

    aois = t6_aois(record)
    counts = {"negative": 0, "positive": 0, "neutral": 0}
    early_counts = {"negative": 0, "positive": 0, "neutral": 0}
    first_ms: dict[str, float | None] = {"negative": None, "positive": None, "neutral": None}
    label_counts = {"left": 0, "center": 0, "right": 0}
    labels: list[str] = []
    x_values: list[float] = []
    screen_valid_points = 0
    aoi_points = 0
    early_aoi_points = 0
    window_rows = eye_rows[start_idx + 1 : end_idx]

    for gaze_row in window_rows:
        ts = row_timestamp(gaze_row)
        point = screen_valid_point(gaze_row)
        if ts is None or point is None:
            continue
        screen_valid_points += 1
        x_values.append(point[0])
        aoi = matched_aoi(point, aois)
        if aoi is None:
            continue
        labels.append(aoi.label)
        label_counts[aoi.label] += 1
        if aoi.emotion not in counts:
            continue
        aoi_points += 1
        counts[aoi.emotion] += 1
        if first_ms[aoi.emotion] is None:
            first_ms[aoi.emotion] = (ts - start_ts) / TIMESTAMP_UNITS_PER_MS
        if 0 <= ts - start_ts <= EARLY_WINDOW_NS:
            early_aoi_points += 1
            early_counts[aoi.emotion] += 1

    neg_first = first_ms["negative"]
    pos_first = first_ms["positive"]
    neg_advantage = None if neg_first is None or pos_first is None else pos_first - neg_first
    pos_delay = None if neg_first is None or pos_first is None else pos_first - neg_first
    transitions = sum(1 for prev, cur in zip(labels, labels[1:]) if prev != cur)
    return {
        **base,
        "cut_status": "cut",
        "window_total_samples": len(window_rows),
        "screen_valid_points": screen_valid_points,
        "aoi_points": aoi_points,
        "negative_points": counts["negative"],
        "positive_points": counts["positive"],
        "neutral_points": counts["neutral"],
        "negative_aoi_ratio": ratio(counts["negative"], aoi_points),
        "positive_aoi_ratio": ratio(counts["positive"], aoi_points),
        "negative_minus_positive_aoi_ratio": f"{(counts['negative'] - counts['positive']) / aoi_points:.6f}" if aoi_points else "0.000000",
        "positive_minus_negative_aoi_ratio": f"{(counts['positive'] - counts['negative']) / aoi_points:.6f}" if aoi_points else "0.000000",
        "first_negative_entry_ms": format_float(neg_first, 3),
        "first_positive_entry_ms": format_float(pos_first, 3),
        "negative_first_entry_advantage_ms": format_float(neg_advantage, 3),
        "positive_first_entry_delay_ms": format_float(pos_delay, 3),
        "early_500ms_aoi_points": early_aoi_points,
        "early_500ms_negative_points": early_counts["negative"],
        "early_500ms_positive_points": early_counts["positive"],
        "early_500ms_negative_ratio": ratio(early_counts["negative"], early_aoi_points),
        "early_500ms_positive_ratio": ratio(early_counts["positive"], early_aoi_points),
        "early_500ms_negative_minus_positive_ratio": f"{(early_counts['negative'] - early_counts['positive']) / early_aoi_points:.6f}" if early_aoi_points else "0.000000",
        "early_500ms_positive_minus_negative_ratio": f"{(early_counts['positive'] - early_counts['negative']) / early_aoi_points:.6f}" if early_aoi_points else "0.000000",
        "t6_left_aoi_ratio": ratio(label_counts["left"], aoi_points),
        "t6_center_aoi_ratio": ratio(label_counts["center"], aoi_points),
        "t6_right_aoi_ratio": ratio(label_counts["right"], aoi_points),
        "t6_aoi_coverage_count": sum(1 for count in label_counts.values() if count > 0),
        "t6_aoi_transition_count": transitions,
        "t6_gaze_x_sd": f"{sd(x_values):.6f}",
        "t6_gaze_x_range": f"{value_range(x_values):.6f}",
    }


def summarize_t5_trial(pair: RawPair, record: dict[str, str], eye_rows: list[dict[str, str]], indices: dict[int, int]) -> dict[str, object]:
    trial = int(float(record["试次轮次"]))
    emotion = normalize_emotion(record["刺激图片情绪类型"])
    start_trigger = trial * 100 + 1
    end_trigger = trial * 100 + 2
    start_idx = indices.get(start_trigger)
    end_idx = indices.get(end_trigger)
    base = {
        "task_type": "T5",
        "subject": pair.subject,
        "subject_id": pair.subject_id,
        "trial": trial,
        "emotion": emotion,
        "start_trigger": start_trigger,
        "end_trigger": end_trigger,
        "cut_status": "missing_trigger_window",
        "raw_records_path": str(pair.records_path),
        "raw_eyetrack_path": pair.eyetrack_path_label,
    }
    if start_idx is None or end_idx is None or start_idx >= end_idx:
        return {**base, **empty_t5_metrics()}
    start_ts = row_timestamp(eye_rows[start_idx])
    end_ts = row_timestamp(eye_rows[end_idx])
    if start_ts is None or end_ts is None:
        return {**base, **empty_t5_metrics()}

    rect = t5_rect(record)
    midpoint = start_ts + (end_ts - start_ts) / 2
    last_start = end_ts - LAST_WINDOW_NS
    window_rows = eye_rows[start_idx + 1 : end_idx]
    screen_valid_points = 0
    image_valid_points = 0
    late_screen = 0
    late_image = 0
    last_screen = 0
    last_image = 0
    leave_count = 0
    revisit_count = 0
    previous_in_image: bool | None = None
    flags: list[bool] = []
    timestamps: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    cells: list[tuple[int, int]] = []
    points: list[tuple[float, float]] = []

    for gaze_row in window_rows:
        ts = row_timestamp(gaze_row)
        point = screen_valid_point(gaze_row)
        if ts is None or point is None:
            continue
        in_image = rect.contains(*point)
        screen_valid_points += 1
        image_valid_points += int(in_image)
        if ts >= midpoint:
            late_screen += 1
            late_image += int(in_image)
        if ts >= last_start:
            last_screen += 1
            last_image += int(in_image)
        if previous_in_image is True and not in_image:
            leave_count += 1
        if previous_in_image is False and in_image:
            revisit_count += 1
        previous_in_image = in_image
        flags.append(in_image)
        timestamps.append(ts)
        if not in_image:
            continue
        nx = min(max((point[0] - rect.x) / rect.w, 0.0), 1.0)
        ny = min(max((point[1] - rect.y) / rect.h, 0.0), 1.0)
        xs.append(nx)
        ys.append(ny)
        cell = grid_cell(nx, ny)
        cells.append(cell)
        points.append((nx, ny))

    durations = run_durations_ms(flags, timestamps)
    x_sd = sd(xs)
    y_sd = sd(ys)
    width = value_range(xs)
    height = value_range(ys)
    return {
        **base,
        "cut_status": "cut",
        "window_total_samples": len(window_rows),
        "screen_valid_points": screen_valid_points,
        "image_valid_points": image_valid_points,
        "valid_viewing_ratio": ratio(image_valid_points, screen_valid_points),
        "late_half_viewing_ratio": ratio(late_image, late_screen),
        "last_1000ms_viewing_ratio": ratio(last_image, last_screen),
        "mean_continuous_viewing_run_ms": f"{mean(durations):.3f}" if durations else "",
        "leave_count": leave_count,
        "revisit_count": revisit_count,
        "gaze_x_sd": f"{x_sd:.6f}",
        "gaze_y_sd": f"{y_sd:.6f}",
        "gaze_xy_dispersion": f"{math.sqrt(x_sd * x_sd + y_sd * y_sd):.6f}",
        "normalized_coverage_width": f"{width:.6f}",
        "normalized_coverage_height": f"{height:.6f}",
        "normalized_coverage_area": f"{width * height:.6f}",
        "grid_coverage_ratio": ratio(len(set(cells)), GRID_CELL_COUNT),
        "grid_entropy": f"{entropy_norm(cells):.6f}",
        **hotspot_metrics(points),
    }


def load_n80_subjects(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    subjects: dict[str, dict[str, str]] = {}
    for row in rows:
        subject_id = row.get("subject_id", "").strip()
        if subject_id:
            subjects[subject_id] = row
    return subjects


def subject_from_path(path: Path) -> tuple[str, str] | None:
    for part in reversed(path.parts):
        if part.startswith("._"):
            continue
        match = re.match(r"^(\d{10,})_", part)
        if match:
            return part, match.group(1)
    for part in reversed(path.parts):
        if part.startswith("._"):
            continue
        match = re.match(r"^(\d{10,})$", part)
        if match:
            return part, match.group(1)
    return None


def task_from_path(path: Path) -> str | None:
    text = str(path)
    if "T5_EmotionalOrientingTask" in text or "/单图/t5/" in text:
        return "T5"
    if "T6_EmotionalObservationTask" in text or "/多图/t6/" in text:
        return "T6"
    return None


def default_raw_roots(repo_root: Path) -> list[tuple[Path, int, str]]:
    return [
        (repo_root / "single_pic" / "单图" / "t5", 0, "workspace_current_t5"),
        (repo_root / "multipic_eye" / "多图" / "t6", 0, "workspace_current_t6"),
        (repo_root / "single_pic" / "lagecy" / "eye_analysis_1.1" / "data", 1, "workspace_legacy_t5t6_eye_analysis_1.1"),
        (repo_root / "single_pic" / "lagecy" / "root_archive", 2, "workspace_legacy_root_archive"),
        (repo_root / "single_pic" / "lagecy" / "eye_analysis_v1_2", 3, "workspace_legacy_eye_analysis_v1_2"),
    ]


def discover_raw_pairs(repo_root: Path, extra_roots: list[Path]) -> list[RawPair]:
    root_specs = default_raw_roots(repo_root)
    root_specs.extend((root, 10 + idx, f"extra_root_{idx + 1}") for idx, root in enumerate(extra_roots))
    candidates: list[RawPair] = []
    for root, rank, label in root_specs:
        if not root.exists():
            continue
        for records_path in [*root.rglob("formal_records.csv"), *root.rglob("formal_recording.csv")]:
            if records_path.name.startswith("._"):
                continue
            eyetrack_paths = eyetrack_files_for_records(records_path)
            if not eyetrack_paths:
                continue
            subject = subject_from_path(records_path)
            task = task_from_path(records_path)
            if subject is None or task is None:
                continue
            subject_name, subject_id = subject
            candidates.append(
                RawPair(
                    task_type=task,
                    subject=subject_name,
                    subject_id=subject_id,
                    records_path=records_path,
                    eyetrack_paths=eyetrack_paths,
                    source_rank=rank,
                    source_label=label,
                )
            )
    return candidates


def eyetrack_files_for_records(records_path: Path) -> tuple[Path, ...]:
    single = records_path.with_name("formal_eyetrack.csv")
    if single.exists() and not single.name.startswith("._"):
        return (single,)
    split_files = tuple(
        sorted(
            path
            for path in records_path.parent.glob("formal_*_eyetrack.csv")
            if not path.name.startswith("._")
        )
    )
    return split_files


def choose_best_pairs(pairs: list[RawPair], n80_ids: set[str]) -> dict[tuple[str, str], RawPair]:
    selected: dict[tuple[str, str], RawPair] = {}
    for pair in sorted(pairs, key=lambda p: (p.source_rank, len(str(p.records_path)), str(p.records_path))):
        if pair.subject_id not in n80_ids:
            continue
        key = (pair.subject_id, pair.task_type)
        selected.setdefault(key, pair)
    return selected


def build_coverage_rows(n80_subjects: dict[str, dict[str, str]], selected: dict[tuple[str, str], RawPair], all_pairs: list[RawPair]) -> list[dict[str, object]]:
    duplicate_counts = Counter((pair.subject_id, pair.task_type) for pair in all_pairs if pair.subject_id in n80_subjects)
    rows: list[dict[str, object]] = []
    for subject_id, n80_row in sorted(n80_subjects.items()):
        row: dict[str, object] = {
            "subject_id": subject_id,
            "subject_name": n80_row.get("subject_name", ""),
            "n_subjects_total": len(n80_subjects),
        }
        for task in ["T5", "T6"]:
            pair = selected.get((subject_id, task))
            prefix = task.lower()
            row[f"{prefix}_raw_status"] = "available" if pair else "missing_raw_pair"
            row[f"{prefix}_raw_subject_folder"] = "" if pair is None else pair.subject
            row[f"{prefix}_raw_source_label"] = "" if pair is None else pair.source_label
            row[f"{prefix}_raw_records_path"] = "" if pair is None else str(pair.records_path)
            row[f"{prefix}_raw_eyetrack_path"] = "" if pair is None else pair.eyetrack_path_label
            row[f"{prefix}_candidate_pair_count"] = duplicate_counts.get((subject_id, task), 0)
        rows.append(row)
    return rows


def process_t6_pair(pair: RawPair, limit_trials: int | None = None) -> list[dict[str, object]]:
    records = read_csv(pair.records_path)
    eye_rows = read_many_csv(pair.eyetrack_paths)
    indices = trigger_indices(eye_rows)
    rows: list[dict[str, object]] = []
    for record in records[:limit_trials]:
        try:
            rows.append(summarize_t6_trial(pair, record, eye_rows, indices))
        except (KeyError, ValueError, SyntaxError):
            continue
    return rows


def process_t5_pair(pair: RawPair, limit_trials: int | None = None) -> list[dict[str, object]]:
    records = read_csv(pair.records_path)
    eye_rows = read_many_csv(pair.eyetrack_paths)
    indices = trigger_indices(eye_rows)
    rows: list[dict[str, object]] = []
    for record in records[:limit_trials]:
        try:
            rows.append(summarize_t5_trial(pair, record, eye_rows, indices))
        except (KeyError, ValueError, SyntaxError):
            continue
    return rows


def summarize_t6_subjects(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in trial_rows:
        grouped.setdefault(str(row["subject_id"]), []).append(row)

    output: list[dict[str, object]] = []
    for subject_id, rows in sorted(grouped.items()):
        valid_rows = [row for row in rows if row["cut_status"] == "cut" and int(row["aoi_points"]) > 0]
        negative_points = sum_int(valid_rows, "negative_points")
        positive_points = sum_int(valid_rows, "positive_points")
        neutral_points = sum_int(valid_rows, "neutral_points")
        aoi_points = sum_int(valid_rows, "aoi_points")
        early_aoi_points = sum_int(valid_rows, "early_500ms_aoi_points")
        early_negative = sum_int(valid_rows, "early_500ms_negative_points")
        early_positive = sum_int(valid_rows, "early_500ms_positive_points")
        output.append(
            {
                "subject": rows[0]["subject"],
                "subject_id": subject_id,
                "n_subjects_total": "",
                "t6_valid_trial_count": len(valid_rows),
                "t6_aoi_points": aoi_points,
                "t6_negative_points": negative_points,
                "t6_positive_points": positive_points,
                "t6_neutral_points": neutral_points,
                "negative_aoi_ratio": ratio(negative_points, aoi_points),
                "positive_aoi_ratio": ratio(positive_points, aoi_points),
                "negative_minus_positive_aoi_ratio": f"{(negative_points - positive_points) / aoi_points:.6f}" if aoi_points else "0.000000",
                "positive_minus_negative_aoi_ratio": f"{(positive_points - negative_points) / aoi_points:.6f}" if aoi_points else "0.000000",
                "mean_first_negative_entry_ms": mean_float(valid_rows, "first_negative_entry_ms"),
                "mean_first_positive_entry_ms": mean_float(valid_rows, "first_positive_entry_ms"),
                "mean_negative_first_entry_advantage_ms": mean_float(valid_rows, "negative_first_entry_advantage_ms"),
                "mean_positive_first_entry_delay_ms": mean_float(valid_rows, "positive_first_entry_delay_ms"),
                "early_500ms_negative_ratio": ratio(early_negative, early_aoi_points),
                "early_500ms_positive_ratio": ratio(early_positive, early_aoi_points),
                "early_500ms_negative_minus_positive_ratio": f"{(early_negative - early_positive) / early_aoi_points:.6f}" if early_aoi_points else "0.000000",
                "early_500ms_positive_minus_negative_ratio": f"{(early_positive - early_negative) / early_aoi_points:.6f}" if early_aoi_points else "0.000000",
                "t6_aoi_coverage_count": mean_float(valid_rows, "t6_aoi_coverage_count"),
                "t6_aoi_transition_count": mean_float(valid_rows, "t6_aoi_transition_count"),
                "t6_gaze_x_sd": mean_float(valid_rows, "t6_gaze_x_sd"),
                "t6_gaze_x_range": mean_float(valid_rows, "t6_gaze_x_range"),
                "missing_negative_entry_trial_count": sum(1 for row in valid_rows if row["first_negative_entry_ms"] == ""),
                "missing_positive_entry_trial_count": sum(1 for row in valid_rows if row["first_positive_entry_ms"] == ""),
            }
        )
    return output


def summarize_t5_subjects(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in trial_rows:
        grouped.setdefault(str(row["subject_id"]), []).append(row)

    output: list[dict[str, object]] = []
    metric_fields = [
        "valid_viewing_ratio",
        "late_half_viewing_ratio",
        "last_1000ms_viewing_ratio",
        "mean_continuous_viewing_run_ms",
        "leave_count",
        "revisit_count",
        "gaze_x_sd",
        "gaze_y_sd",
        "gaze_xy_dispersion",
        "normalized_coverage_area",
        "grid_coverage_ratio",
        "grid_entropy",
        "hotspot_top_cell_ratio",
        "hotspot_top_5pct_cells_ratio",
        "hotspot_top_10pct_cells_ratio",
        "gaze_to_hotspot_mean_distance",
        "gaze_to_hotspot_median_distance",
    ]
    for subject_id, rows in sorted(grouped.items()):
        row_out: dict[str, object] = {"subject": rows[0]["subject"], "subject_id": subject_id, "n_subjects_total": ""}
        for emotion in ["positive", "negative", "neutral", "complex"]:
            subset = [
                row
                for row in rows
                if row["emotion"] == emotion and row["cut_status"] == "cut" and int(row["screen_valid_points"]) > 0
            ]
            prefix = f"{emotion}_"
            row_out[f"{prefix}valid_trial_count"] = len(subset)
            row_out[f"{prefix}image_valid_points"] = sum_int(subset, "image_valid_points")
            row_out[f"{prefix}screen_valid_points"] = sum_int(subset, "screen_valid_points")
            for field in metric_fields:
                row_out[f"{prefix}{field}"] = mean_float(subset, field)
        row_out["negative_minus_positive_late_half_ratio"] = diff_string(str(row_out["negative_late_half_viewing_ratio"]), str(row_out["positive_late_half_viewing_ratio"]))
        row_out["negative_minus_positive_last_1000ms_ratio"] = diff_string(str(row_out["negative_last_1000ms_viewing_ratio"]), str(row_out["positive_last_1000ms_viewing_ratio"]))
        row_out["negative_minus_positive_run_ms"] = diff_string(str(row_out["negative_mean_continuous_viewing_run_ms"]), str(row_out["positive_mean_continuous_viewing_run_ms"]))
        row_out["negative_minus_positive_revisit_count"] = diff_string(str(row_out["negative_revisit_count"]), str(row_out["positive_revisit_count"]))
        row_out["negative_minus_positive_top_cell_ratio"] = diff_string(str(row_out["negative_hotspot_top_cell_ratio"]), str(row_out["positive_hotspot_top_cell_ratio"]))
        row_out["negative_minus_positive_top_10pct_cells_ratio"] = diff_string(str(row_out["negative_hotspot_top_10pct_cells_ratio"]), str(row_out["positive_hotspot_top_10pct_cells_ratio"]))
        row_out["negative_minus_positive_hotspot_distance"] = diff_string(str(row_out["negative_gaze_to_hotspot_mean_distance"]), str(row_out["positive_gaze_to_hotspot_mean_distance"]))
        emotion_entropy_values = [str(row_out[f"{emotion}_grid_entropy"]) for emotion in ["positive", "negative", "neutral"]]
        emotion_entropy = mean_available(emotion_entropy_values)
        row_out["emotion_mean_grid_entropy"] = emotion_entropy
        row_out["complex_minus_emotion_grid_entropy"] = diff_string(str(row_out["complex_grid_entropy"]), emotion_entropy)
        output.append(row_out)
    return output


def mean_available(values: list[str]) -> str:
    nums = [float(value) for value in values if value != ""]
    return f"{mean(nums):.6f}" if nums else ""


BASE_TRIAL_FIELDS = [
    "task_type",
    "subject",
    "subject_id",
    "trial",
    "emotion",
    "start_trigger",
    "end_trigger",
    "cut_status",
    "raw_records_path",
    "raw_eyetrack_path",
]

T6_TRIAL_FIELDS = [
    *BASE_TRIAL_FIELDS,
    *empty_t6_metrics().keys(),
]

T5_TRIAL_FIELDS = [
    *BASE_TRIAL_FIELDS,
    *empty_t5_metrics().keys(),
]

T6_SUBJECT_FIELDS = [
    "subject",
    "subject_id",
    "n_subjects_total",
    "t6_valid_trial_count",
    "t6_aoi_points",
    "t6_negative_points",
    "t6_positive_points",
    "t6_neutral_points",
    "negative_aoi_ratio",
    "positive_aoi_ratio",
    "negative_minus_positive_aoi_ratio",
    "positive_minus_negative_aoi_ratio",
    "mean_first_negative_entry_ms",
    "mean_first_positive_entry_ms",
    "mean_negative_first_entry_advantage_ms",
    "mean_positive_first_entry_delay_ms",
    "early_500ms_negative_ratio",
    "early_500ms_positive_ratio",
    "early_500ms_negative_minus_positive_ratio",
    "early_500ms_positive_minus_negative_ratio",
    "t6_aoi_coverage_count",
    "t6_aoi_transition_count",
    "t6_gaze_x_sd",
    "t6_gaze_x_range",
    "missing_negative_entry_trial_count",
    "missing_positive_entry_trial_count",
]

T5_SUBJECT_FIELDS = ["subject", "subject_id", "n_subjects_total"]
for _emotion in ["positive", "negative", "neutral", "complex"]:
    T5_SUBJECT_FIELDS.extend(
        [
            f"{_emotion}_valid_trial_count",
            f"{_emotion}_image_valid_points",
            f"{_emotion}_screen_valid_points",
            f"{_emotion}_valid_viewing_ratio",
            f"{_emotion}_late_half_viewing_ratio",
            f"{_emotion}_last_1000ms_viewing_ratio",
            f"{_emotion}_mean_continuous_viewing_run_ms",
            f"{_emotion}_leave_count",
            f"{_emotion}_revisit_count",
            f"{_emotion}_gaze_x_sd",
            f"{_emotion}_gaze_y_sd",
            f"{_emotion}_gaze_xy_dispersion",
            f"{_emotion}_normalized_coverage_area",
            f"{_emotion}_grid_coverage_ratio",
            f"{_emotion}_grid_entropy",
            f"{_emotion}_hotspot_top_cell_ratio",
            f"{_emotion}_hotspot_top_5pct_cells_ratio",
            f"{_emotion}_hotspot_top_10pct_cells_ratio",
            f"{_emotion}_gaze_to_hotspot_mean_distance",
            f"{_emotion}_gaze_to_hotspot_median_distance",
        ]
    )
T5_SUBJECT_FIELDS.extend(
    [
        "negative_minus_positive_late_half_ratio",
        "negative_minus_positive_last_1000ms_ratio",
        "negative_minus_positive_run_ms",
        "negative_minus_positive_revisit_count",
        "negative_minus_positive_top_cell_ratio",
        "negative_minus_positive_top_10pct_cells_ratio",
        "negative_minus_positive_hotspot_distance",
        "emotion_mean_grid_entropy",
        "complex_minus_emotion_grid_entropy",
    ]
)

COVERAGE_FIELDS = [
    "subject_id",
    "subject_name",
    "n_subjects_total",
    "t5_raw_status",
    "t5_raw_subject_folder",
    "t5_raw_source_label",
    "t5_raw_records_path",
    "t5_raw_eyetrack_path",
    "t5_candidate_pair_count",
    "t6_raw_status",
    "t6_raw_subject_folder",
    "t6_raw_source_label",
    "t6_raw_records_path",
    "t6_raw_eyetrack_path",
    "t6_candidate_pair_count",
]


def attach_total(rows: list[dict[str, object]], n_total: int) -> list[dict[str, object]]:
    for row in rows:
        row["n_subjects_total"] = n_total
    return rows


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[5]
    default_n80 = repo_root / "t5t6 联合数据分析" / "260616" / "data" / "单图多图参数_量表匹配总表_合并_N80.csv"
    default_output = script_path.parent / "raw重算_direct_model_free_N80"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n80-table", type=Path, default=default_n80, help="N=80 subject-frame CSV.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Output directory for rebuilt direct model-free CSVs.")
    parser.add_argument("--extra-raw-root", type=Path, action="append", default=[], help="Additional raw root to search recursively.")
    parser.add_argument("--coverage-only", action="store_true", help="Only discover raw pairs and write coverage table.")
    parser.add_argument("--limit-subjects", type=int, default=None, help="Process only the first N selected subject ids for smoke tests.")
    parser.add_argument("--limit-trials", type=int, default=None, help="Process only the first N records per raw pair for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[5]
    n80_subjects = load_n80_subjects(args.n80_table)
    n80_ids = set(n80_subjects)
    all_pairs = discover_raw_pairs(repo_root, args.extra_raw_root)
    selected = choose_best_pairs(all_pairs, n80_ids)
    if args.limit_subjects is not None:
        keep_ids = set(sorted(n80_ids)[: args.limit_subjects])
        selected = {key: pair for key, pair in selected.items() if key[0] in keep_ids}

    coverage_rows = build_coverage_rows(n80_subjects, selected, all_pairs)
    write_csv(args.output_dir / "raw_direct_coverage_N80.csv", COVERAGE_FIELDS, coverage_rows)
    if args.coverage_only:
        print(f"n_subjects_total={len(n80_subjects)}")
        print(f"t5_available={sum(1 for row in coverage_rows if row['t5_raw_status'] == 'available')}")
        print(f"t6_available={sum(1 for row in coverage_rows if row['t6_raw_status'] == 'available')}")
        print(args.output_dir)
        return

    t6_trial_rows: list[dict[str, object]] = []
    t5_trial_rows: list[dict[str, object]] = []
    for (subject_id, task_type), pair in sorted(selected.items()):
        if task_type == "T6":
            t6_trial_rows.extend(process_t6_pair(pair, args.limit_trials))
        elif task_type == "T5":
            t5_trial_rows.extend(process_t5_pair(pair, args.limit_trials))

    t6_subject_rows = attach_total(summarize_t6_subjects(t6_trial_rows), len(n80_subjects))
    t5_subject_rows = attach_total(summarize_t5_subjects(t5_trial_rows), len(n80_subjects))
    write_csv(args.output_dir / "T6_raw_direct_trial_N80.csv", T6_TRIAL_FIELDS, t6_trial_rows)
    write_csv(args.output_dir / "T6_raw_direct_subject_N80.csv", T6_SUBJECT_FIELDS, t6_subject_rows)
    write_csv(args.output_dir / "T5_raw_direct_trial_N80.csv", T5_TRIAL_FIELDS, t5_trial_rows)
    write_csv(args.output_dir / "T5_raw_direct_subject_N80.csv", T5_SUBJECT_FIELDS, t5_subject_rows)

    print(f"n_subjects_total={len(n80_subjects)}")
    print(f"t5_available={sum(1 for row in coverage_rows if row['t5_raw_status'] == 'available')}")
    print(f"t6_available={sum(1 for row in coverage_rows if row['t6_raw_status'] == 'available')}")
    print(f"t5_trial_rows={len(t5_trial_rows)}")
    print(f"t5_subject_rows={len(t5_subject_rows)}")
    print(f"t6_trial_rows={len(t6_trial_rows)}")
    print(f"t6_subject_rows={len(t6_subject_rows)}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
