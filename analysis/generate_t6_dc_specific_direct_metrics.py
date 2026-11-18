#!/usr/bin/env python3
"""Generate subject-level direct spatial metrics for T6 R03c dc validation."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PARAM_CSV = SCRIPT_DIR.parents[2] / "data" / "单图多图参数_量表匹配总表_合并_N80.csv"
R03B_DIR = SCRIPT_DIR / "core_model_free_tmax_stability_R03b_N80"
PATH_CSV = R03B_DIR / "T6_Tmax_production_reproduction_subject_N80.csv"
OUTPUT_DIR = SCRIPT_DIR / "core_model_free_dc_specific_validation_R03c_N80"
OUTPUT_CSV = OUTPUT_DIR / "dc_specific_direct_metrics_N80.csv"

SCREEN_CENTER = np.array([960.0, 540.0])
SCREEN_WIDTH = 1920.0
SCREEN_HEIGHT = 1080.0
TIMESTAMP_UNITS_PER_MS = 1_000_000.0
CENTER_RADII = (75.0, 100.0, 150.0)
EARLY_WINDOWS_MS = (300.0, 500.0, 800.0)
MIN_CENTER_POINTS = 5
MIN_ENDPOINT_RUN = 5


@dataclass(frozen=True)
class Rect:
    label: str
    emotion: str
    x: float
    y: float
    w: float
    h: float

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x + self.w / 2.0, self.y + self.h / 2.0])


def parse_pair(value: object) -> tuple[float, float]:
    parsed = ast.literal_eval(str(value))
    return float(parsed[0]), float(parsed[1])


def normalize_emotion(value: object) -> str:
    normalized = str(value).strip().lower()
    return {"postive": "positive", "nagetive": "negative"}.get(normalized, normalized)


def trial_aois(row: pd.Series) -> list[Rect]:
    specs = (("left", "左"), ("center", "中"), ("right", "右"))
    output: list[Rect] = []
    for label, prefix in specs:
        x, y = parse_pair(row[f"{prefix}图片坐标"])
        w, h = parse_pair(row[f"{prefix}图片大小"])
        output.append(Rect(label, normalize_emotion(row[f"{prefix}情绪类型"]), x, y, w, h))
    return output


def find_aoi(point: np.ndarray, aois: list[Rect]) -> Rect | None:
    for aoi in aois:
        if aoi.contains(float(point[0]), float(point[1])):
            return aoi
    return None


def rms_about_centroid(points: list[np.ndarray]) -> float:
    array = np.asarray(points, dtype=float)
    centroid = array.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((array - centroid) ** 2, axis=1))))


def raw_attention(time_ms: float, pmax: float, tmax: float) -> float:
    if time_ms < 0 or tmax <= 0:
        return 0.0
    return float(pmax * (time_ms / tmax) * math.exp(1.0 - time_ms / tmax))


def expected_position_error(
    point: np.ndarray,
    time_ms: float,
    aois: list[Rect],
    params: dict[str, float],
) -> float:
    centers = {aoi.emotion: aoi.center for aoi in aois}
    weights = {"central": 1.0}
    for emotion in ("positive", "negative", "neutral"):
        weights[emotion] = raw_attention(
            time_ms,
            params[f"Pmax_{emotion}_多图"],
            params[f"Tmax_{emotion}_多图"],
        )
    total = sum(weights.values())
    expected = weights["central"] * SCREEN_CENTER
    for emotion in ("positive", "negative", "neutral"):
        expected = expected + weights[emotion] * centers[emotion]
    expected = expected / total
    return float(np.linalg.norm(point - expected))


def trigger_positions(frame: pd.DataFrame) -> dict[int, int]:
    result: dict[int, int] = {}
    trigger = pd.to_numeric(frame["trigger"], errors="coerce")
    for index, value in trigger.items():
        if pd.notna(value) and int(value) > 0 and int(value) not in result:
            result[int(value)] = int(index)
    return result


def summarize_trial(
    trial: int,
    eye: pd.DataFrame,
    trigger_map: dict[int, int],
    record: pd.Series,
    params: dict[str, float],
) -> dict[str, float] | None:
    start_index = trigger_map.get(trial * 100 + 1)
    end_index = trigger_map.get(trial * 100 + 2)
    if start_index is None or end_index is None or start_index >= end_index:
        return None
    start_ts = eye.loc[start_index, "timestamp"]
    if pd.isna(start_ts):
        return None
    aois = trial_aois(record)
    samples: list[dict[str, object]] = []
    total = 0
    bino_valid = 0
    screen_valid = 0
    window = eye.loc[start_index + 1 : end_index - 1]
    for valid, x, y, ts in zip(
        window["bino_eye_valid"].to_numpy(),
        window["bino_eye_gaze_position_x"].to_numpy(),
        window["bino_eye_gaze_position_y"].to_numpy(),
        window["timestamp"].to_numpy(),
    ):
        total += 1
        if pd.notna(valid) and int(valid) == 1:
            bino_valid += 1
        point = None
        if (
            pd.notna(valid) and int(valid) == 1 and pd.notna(x) and pd.notna(y)
            and 0.0 <= float(x) <= SCREEN_WIDTH and 0.0 <= float(y) <= SCREEN_HEIGHT
        ):
            point = np.array([float(x), float(y)])
        if point is not None:
            screen_valid += 1
        if point is None or pd.isna(ts):
            samples.append({"point": None, "time_ms": math.nan, "aoi": None})
            continue
        time_ms = (float(ts) - float(start_ts)) / TIMESTAMP_UNITS_PER_MS
        samples.append({"point": point, "time_ms": time_ms, "aoi": find_aoi(point, aois)})

    valid_samples = [sample for sample in samples if sample["point"] is not None]
    result: dict[str, float] = {
        "trial": float(trial),
        "window_total_samples": float(total),
        "bino_valid_points": float(bino_valid),
        "screen_valid_points": float(screen_valid),
        "missing_points": float(total - bino_valid),
        "out_of_screen_points": float(max(bino_valid - screen_valid, 0)),
    }
    for radius in CENTER_RADII:
        prefix: list[np.ndarray] = []
        exited = False
        for sample in samples:
            point = sample["point"]
            if point is None:
                continue
            distance = float(np.linalg.norm(point - SCREEN_CENTER))
            if distance > radius:
                exited = True
                break
            prefix.append(point)
        suffix = int(radius)
        result[f"center_gaze_dispersion_r{suffix}_px"] = (
            rms_about_centroid(prefix) if len(prefix) >= MIN_CENTER_POINTS else math.nan
        )
        result[f"mean_distance_to_center_r{suffix}_px"] = (
            float(np.mean([np.linalg.norm(point - SCREEN_CENTER) for point in prefix]))
            if len(prefix) >= MIN_CENTER_POINTS
            else math.nan
        )
        endpoint_error = math.nan
        if exited:
            exit_position = next(
                (i for i, sample in enumerate(samples) if sample["point"] is not None and np.linalg.norm(sample["point"] - SCREEN_CENTER) > radius),
                None,
            )
            if exit_position is not None:
                run: list[np.ndarray] = []
                run_aoi: Rect | None = None
                for sample in samples[exit_position:]:
                    point = sample["point"]
                    aoi = sample["aoi"]
                    if point is None or aoi is None:
                        run = []
                        run_aoi = None
                        continue
                    if run_aoi is None or aoi.label != run_aoi.label:
                        run = [point]
                        run_aoi = aoi
                    else:
                        run.append(point)
                    if len(run) >= MIN_ENDPOINT_RUN and run_aoi is not None:
                        endpoint = np.median(np.asarray(run[:MIN_ENDPOINT_RUN]), axis=0)
                        endpoint_error = float(np.linalg.norm(endpoint - run_aoi.center))
                        break
        result[f"transition_endpoint_dispersion_r{suffix}_px"] = endpoint_error

    for window in EARLY_WINDOWS_MS:
        distances = [
            float(np.linalg.norm(sample["point"] - SCREEN_CENTER))
            for sample in valid_samples
            if 0.0 <= float(sample["time_ms"]) <= window
        ]
        result[f"early_phase_center_escape_radius_{int(window)}ms_px"] = (
            float(np.percentile(distances, 95)) if distances else math.nan
        )

    within_values: list[float] = []
    for aoi in aois:
        points = [sample["point"] for sample in valid_samples if sample["aoi"] is not None and sample["aoi"].label == aoi.label]
        if len(points) >= MIN_CENTER_POINTS:
            within_values.append(rms_about_centroid(points))
    result["within_AOI_gaze_dispersion_px"] = float(np.mean(within_values)) if within_values else math.nan
    residuals = [
        expected_position_error(sample["point"], float(sample["time_ms"]), aois, params)
        for sample in valid_samples
        if float(sample["time_ms"]) >= 0.0
    ]
    result["residual_spatial_error_px"] = float(np.mean(residuals)) if residuals else math.nan
    return result


def aggregate(trials: list[dict[str, float]], split: str) -> dict[str, float]:
    if split == "odd":
        selected = [row for row in trials if int(row["trial"]) % 2 == 1]
    elif split == "even":
        selected = [row for row in trials if int(row["trial"]) % 2 == 0]
    else:
        selected = trials
    suffix = "" if split == "all" else f"_{split}"
    output: dict[str, float] = {f"valid_trial_count{suffix}": float(len(selected))}
    count_fields = ("window_total_samples", "bino_valid_points", "screen_valid_points", "missing_points", "out_of_screen_points")
    for field in count_fields:
        output[f"{field}{suffix}"] = float(sum(row[field] for row in selected))
    total = output[f"window_total_samples{suffix}"]
    bino = output[f"bino_valid_points{suffix}"]
    output[f"missing_rate{suffix}"] = output[f"missing_points{suffix}"] / total if total else math.nan
    output[f"out_of_screen_rate{suffix}"] = output[f"out_of_screen_points{suffix}"] / bino if bino else math.nan
    metric_fields = [field for field in trials[0] if field not in {"trial", *count_fields}] if trials else []
    for field in metric_fields:
        values = [row[field] for row in selected if pd.notna(row[field])]
        output[f"{field}{suffix}"] = float(np.mean(values)) if values else math.nan
        output[f"{field}_n_trials{suffix}"] = float(len(values))
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    params = pd.read_csv(PARAM_CSV, dtype={"subject_id": str})
    paths = pd.read_csv(PATH_CSV, dtype={"subject_id": str})
    source = params.merge(paths[["subject_id", "raw_eyetrack_path", "raw_records_path"]], on="subject_id", how="inner")
    rows: list[dict[str, object]] = []
    for index, row in source.iterrows():
        subject_id = str(row["subject_id"])
        eye = pd.read_csv(Path(row["raw_eyetrack_path"]), encoding="utf-8-sig")
        records = pd.read_csv(Path(row["raw_records_path"]), encoding="utf-8-sig")
        eye = eye.reset_index(drop=True)
        for column in (
            "timestamp", "trigger", "bino_eye_valid",
            "bino_eye_gaze_position_x", "bino_eye_gaze_position_y",
        ):
            eye[column] = pd.to_numeric(eye[column], errors="coerce")
        triggers = trigger_positions(eye)
        parameter_values = {
            name: float(row[name])
            for name in (
                "Pmax_positive_多图", "Pmax_negative_多图", "Pmax_neutral_多图",
                "Tmax_positive_多图", "Tmax_negative_多图", "Tmax_neutral_多图",
            )
        }
        trial_rows: list[dict[str, float]] = []
        for trial_index, (_, record) in enumerate(records.iloc[:40].iterrows(), start=1):
            trial_number = int(float(record.get("试次轮次", trial_index)))
            summary = summarize_trial(trial_number, eye, triggers, record, parameter_values)
            if summary is not None:
                trial_rows.append(summary)
        subject_row: dict[str, object] = {
            "subject_id": subject_id,
            "raw_eyetrack_path": row["raw_eyetrack_path"],
            "raw_records_path": row["raw_records_path"],
        }
        for split in ("all", "odd", "even"):
            subject_row.update(aggregate(trial_rows, split))
        rename = {
            "center_gaze_dispersion_r100_px": "center_gaze_dispersion_px",
            "mean_distance_to_center_r100_px": "mean_distance_to_center_px",
            "transition_endpoint_dispersion_r100_px": "transition_endpoint_dispersion_px",
            "early_phase_center_escape_radius_500ms_px": "early_phase_center_escape_radius_px",
        }
        for old, new in rename.items():
            subject_row[new] = subject_row.get(old, math.nan)
            subject_row[f"{new}_odd"] = subject_row.get(f"{old}_odd", math.nan)
            subject_row[f"{new}_even"] = subject_row.get(f"{old}_even", math.nan)
        rows.append(subject_row)
        print(f"[{index + 1:02d}/{len(source)}] subject_id={subject_id} trials={len(trial_rows)}")
    output = pd.DataFrame(rows).sort_values("subject_id")
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"rows={len(output)} subjects={output['subject_id'].nunique()} output={OUTPUT_CSV}")


if __name__ == "__main__":
    main()
