import ast
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from corecode.clean_t5_t6_exports import prepare_clean_raw_data
from devide_rule.classify_subject import classify, load_thresholds

DEFAULT_INIT_PARAMS = {
    "Tmax1": 1000.0,
    "Tmax2": 1000.0,
    "Tmax3": 1000.0,
    "Pmax1": 1000.0,
    "Pmax2": 1000.0,
    "Pmax3": 1000.0,
    "dc": 120.0,
}


DEFAULT_BOUNDS = {
    "Tmax1": (1e-3, 100000.0),
    "Tmax2": (1e-3, 100000.0),
    "Tmax3": (1e-3, 100000.0),
    "Pmax1": (0.0, 100000.0),
    "Pmax2": (0.0, 100000.0),
    "Pmax3": (0.0, 100000.0),
    "dc": (1.0, 2000.0),
}

SCREEN_WIDTH_CM = 34.5
SCREEN_HEIGHT_CM = 19.5
SCREEN_WIDTH_PX = 1920.0
SCREEN_HEIGHT_PX = 1080.0
VIEWING_DISTANCE_CM = 57.0


def px_to_visual_angle_deg(
    px,
    screen_cm=SCREEN_WIDTH_CM,
    resolution_px=SCREEN_WIDTH_PX,
    distance_cm=VIEWING_DISTANCE_CM,
):
    cm = px * screen_cm / resolution_px
    return 2 * np.degrees(np.arctan(cm / (2 * distance_cm)))


def parse_pair(value):
    if pd.isna(value):
        return (np.nan, np.nan)
    return ast.literal_eval(str(value).strip())


def normalize_emotion(value):
    if pd.isna(value):
        return ""
    normalized = str(value).strip().lower()
    if normalized == "postive":
        return "positive"
    if normalized == "nagetive":
        return "negative"
    return normalized


def in_rect_mask(gaze_x, gaze_y, left, top, width, height):
    if any(pd.isna(v) for v in [left, top, width, height]):
        return np.zeros(len(gaze_x), dtype=bool)
    return (
        gaze_x.ge(left)
        & gaze_x.le(left + width)
        & gaze_y.ge(top)
        & gaze_y.le(top + height)
    )


def build_base_frame(eyetrack_df):
    required_columns = [
        "timestamp",
        "bino_eye_gaze_position_x",
        "bino_eye_gaze_position_y",
        "bino_eye_valid",
        "trigger",
    ]
    missing = [column for column in required_columns if column not in eyetrack_df.columns]
    if missing:
        raise ValueError(f"眼动数据缺少 T6 双眼必要列: {missing}")

    frame = eyetrack_df[required_columns].copy()
    frame.columns = ["timestamp", "gaze_x", "gaze_y", "gaze_valid", "trigger"]
    for column in ["timestamp", "gaze_x", "gaze_y", "gaze_valid"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trigger"] = pd.to_numeric(frame["trigger"], errors="coerce")
    frame["trial_number"] = 0
    frame["fixation"] = 0
    frame["picture"] = 0
    frame["blank"] = 0

    if 4003 not in frame["trigger"].values:
        raise ValueError("未找到结束 trigger=4003")

    end_task_idx = frame.index[frame["trigger"] == 4003].tolist()[0]
    trial_count = 0

    for trial_num in range(1, 41):
        trig_fix = trial_num * 100
        trig_pic = trig_fix + 1
        trig_blank = trig_fix + 2

        fix_idx_list = frame.index[frame["trigger"] == trig_fix].tolist()
        pic_idx_list = frame.index[frame["trigger"] == trig_pic].tolist()
        blank_idx_list = frame.index[frame["trigger"] == trig_blank].tolist()

        if not fix_idx_list or not pic_idx_list or not blank_idx_list:
            continue

        trial_count += 1
        fix_idx = fix_idx_list[0]
        pic_idx = pic_idx_list[0]
        blank_idx = blank_idx_list[0]

        if trial_num < 40:
            next_fix_trigger = (trial_num + 1) * 100
            next_fix_idx_list = frame.index[frame["trigger"] == next_fix_trigger].tolist()
            blank_end_idx = next_fix_idx_list[0] - 1 if next_fix_idx_list else end_task_idx
        else:
            blank_end_idx = end_task_idx

        frame.loc[fix_idx:pic_idx - 1, "trial_number"] = trial_num
        frame.loc[fix_idx:pic_idx - 1, "fixation"] = 1

        frame.loc[pic_idx:blank_idx - 1, "trial_number"] = trial_num
        frame.loc[pic_idx:blank_idx - 1, "picture"] = 1

        frame.loc[blank_idx:blank_end_idx, "trial_number"] = trial_num
        frame.loc[blank_idx:blank_end_idx, "blank"] = 1

    return frame, trial_count


def annotate_spatial_and_emotion(base_df, records_df):
    frame = base_df.copy()
    frame["left"] = 0
    frame["middle"] = 0
    frame["right"] = 0
    frame["positive"] = 0
    frame["nagetive"] = 0
    frame["neutral"] = 0

    for trial_num, row in enumerate(records_df.itertuples(index=False), start=1):
        trial_mask = frame["trial_number"].eq(trial_num)
        picture_mask = frame["picture"].eq(1)
        sub_mask = trial_mask & picture_mask
        if not sub_mask.any():
            continue

        sub = frame.loc[sub_mask, ["gaze_x", "gaze_y"]]

        left_x, left_y = parse_pair(getattr(row, "左图片坐标"))
        left_w, left_h = parse_pair(getattr(row, "左图片大小"))
        middle_x, middle_y = parse_pair(getattr(row, "中图片坐标"))
        middle_w, middle_h = parse_pair(getattr(row, "中图片大小"))
        right_x, right_y = parse_pair(getattr(row, "右图片坐标"))
        right_w, right_h = parse_pair(getattr(row, "右图片大小"))

        left_hits = in_rect_mask(sub["gaze_x"], sub["gaze_y"], left_x, left_y, left_w, left_h)
        middle_hits = in_rect_mask(sub["gaze_x"], sub["gaze_y"], middle_x, middle_y, middle_w, middle_h)
        right_hits = in_rect_mask(sub["gaze_x"], sub["gaze_y"], right_x, right_y, right_w, right_h)

        sub_index = sub.index
        frame.loc[sub_index[left_hits], "left"] = 1
        frame.loc[sub_index[middle_hits], "middle"] = 1
        frame.loc[sub_index[right_hits], "right"] = 1

        emotion_map = {
            "left": normalize_emotion(getattr(row, "左情绪类型")),
            "middle": normalize_emotion(getattr(row, "中情绪类型")),
            "right": normalize_emotion(getattr(row, "右情绪类型")),
        }

        for side, hits in [("left", left_hits), ("middle", middle_hits), ("right", right_hits)]:
            if not hits.any():
                continue
            emotion = emotion_map[side]
            hit_index = sub_index[hits]
            if emotion == "positive":
                frame.loc[hit_index, "positive"] = 1
            elif emotion == "negative":
                frame.loc[hit_index, "nagetive"] = 1
            elif emotion == "neutral":
                frame.loc[hit_index, "neutral"] = 1

    return frame


def prepare_model_data(frame, timestamp_unit_scale=1e6):
    required_cols = [
        "timestamp",
        "gaze_x",
        "gaze_y",
        "gaze_valid",
        "trial_number",
        "fixation",
        "picture",
        "blank",
        "left",
        "middle",
        "right",
        "positive",
        "nagetive",
        "neutral",
    ]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"数据缺少必要列: {missing}")

    df = frame[required_cols].copy()

    for column in ["timestamp", "gaze_x", "gaze_y", "gaze_valid"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    coord_mask = df["gaze_x"].between(0, 1920) & df["gaze_y"].between(0, 1080)
    validity_mask = df["gaze_valid"].eq(1)
    df = df[coord_mask & validity_mask].copy()
    if df.empty:
        raise ValueError("双眼有效性和坐标筛选后数据为空。")

    df = df.sort_values(["trial_number", "timestamp"]).reset_index(drop=True)

    trial_start = df.groupby("trial_number")["timestamp"].transform("min")
    df["t_from_trial_start"] = (df["timestamp"] - trial_start) / timestamp_unit_scale

    picture_start_map = (
        df[df["picture"] == 1]
        .groupby("trial_number")["timestamp"]
        .min()
        .to_dict()
    )
    df["picture_start_time"] = df["trial_number"].map(picture_start_map)
    df["t_from_picture_start"] = (
        (df["timestamp"] - df["picture_start_time"]) / timestamp_unit_scale
    )

    df = df[(df["fixation"] == 1) | (df["picture"] == 1)].copy()

    zero_emotion_mask = (df["positive"] + df["nagetive"] + df["neutral"]) == 0
    df = df[~((df["picture"] == 1) & zero_emotion_mask)].copy()
    if df.empty:
        raise ValueError("筛选后数据为空。")

    prepared = df[
        [
            "timestamp",
            "gaze_x",
            "gaze_y",
            "trial_number",
            "positive",
            "nagetive",
            "neutral",
            "t_from_trial_start",
            "t_from_picture_start",
        ]
    ].copy()

    data = {
        "timestamp": prepared["timestamp"].to_numpy(dtype=np.float64),
        "gaze_x": prepared["gaze_x"].to_numpy(dtype=np.float64),
        "gaze_y": prepared["gaze_y"].to_numpy(dtype=np.float64),
        "trial_number": prepared["trial_number"].to_numpy(dtype=np.int64),
        "positive": prepared["positive"].to_numpy(dtype=np.float64),
        "nagetive": prepared["nagetive"].to_numpy(dtype=np.float64),
        "neutral": prepared["neutral"].to_numpy(dtype=np.float64),
        "t_from_trial_start": prepared["t_from_trial_start"].to_numpy(dtype=np.float64),
        "t_from_picture_start": prepared["t_from_picture_start"].to_numpy(dtype=np.float64),
    }
    return prepared, data


def compute_log_likelihood(data, params, eps=1e-12):
    x = data["gaze_x"]
    y = data["gaze_y"]
    positive = data["positive"]
    nagetive = data["nagetive"]
    neutral = data["neutral"]
    t_trial = data["t_from_trial_start"]
    t_picture = data["t_from_picture_start"]

    tmax1 = params["Tmax1"]
    tmax2 = params["Tmax2"]
    tmax3 = params["Tmax3"]
    pmax1 = params["Pmax1"]
    pmax2 = params["Pmax2"]
    pmax3 = params["Pmax3"]
    dc = params["dc"]

    if dc <= 0:
        return -np.inf
    if min(tmax1, tmax2, tmax3) <= 0:
        return -np.inf
    if min(pmax1, pmax2, pmax3) < 0:
        return -np.inf

    def raw_attention(time_values, pmax, tmax):
        values = np.zeros_like(time_values, dtype=np.float64)
        valid = time_values >= 0
        values[valid] = (
            pmax
            * (time_values[valid] / (tmax + eps))
            * np.exp(1.0 - time_values[valid] / (tmax + eps))
        )
        return values

    p0_raw = np.ones_like(t_trial, dtype=np.float64)
    p1_raw = raw_attention(t_picture, pmax1, tmax1)
    p2_raw = raw_attention(t_picture, pmax2, tmax2)
    p3_raw = raw_attention(t_picture, pmax3, tmax3)

    p_sum = p0_raw + p1_raw + p2_raw + p3_raw
    p0 = p0_raw / (p_sum + eps)
    p1 = p1_raw / (p_sum + eps)
    p2 = p2_raw / (p_sum + eps)
    p3 = p3_raw / (p_sum + eps)

    x0 = 1920.0 / 2.0
    y0 = 1080.0 / 2.0
    g0 = (1.0 / (2.0 * np.pi * dc ** 2)) * np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * dc ** 2 + eps)
    )

    p_xy_t = p0 * g0 + p1 * positive + p2 * nagetive + p3 * neutral
    p_xy_t = np.clip(p_xy_t, eps, None)
    return float(np.sum(np.log(p_xy_t)))


def optimize_free_parameters(data, init_params=None, bounds=None, method="L-BFGS-B"):
    param_names = ["Tmax1", "Tmax2", "Tmax3", "Pmax1", "Pmax2", "Pmax3", "dc"]
    init = DEFAULT_INIT_PARAMS.copy()
    if init_params:
        init.update(init_params)
    bound_values = DEFAULT_BOUNDS.copy()
    if bounds:
        bound_values.update(bounds)

    x0 = np.array([init[name] for name in param_names], dtype=np.float64)
    bounds_list = [bound_values[name] for name in param_names]

    def objective(theta):
        params = {name: value for name, value in zip(param_names, theta)}
        ll = compute_log_likelihood(data, params)
        if not np.isfinite(ll):
            return 1e100
        return -ll

    result = minimize(objective, x0=x0, bounds=bounds_list, method=method)
    best_params = {name: float(value) for name, value in zip(param_names, result.x)}
    pmax_sum = best_params["Pmax1"] + best_params["Pmax2"] + best_params["Pmax3"]
    pm1 = best_params["Pmax1"] / pmax_sum if pmax_sum else 0.0
    pm2 = best_params["Pmax2"] / pmax_sum if pmax_sum else 0.0
    pm3 = best_params["Pmax3"] / pmax_sum if pmax_sum else 0.0

    return {
        "best_params": best_params,
        "best_log_likelihood": float(-result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "Pm1_positive": float(pm1),
        "Pm2_negative": float(pm2),
        "Pm3_neutral": float(pm3),
        "n_observations": int(len(data["timestamp"])),
        "n_trials_with_records": int(pd.Series(data["trial_number"]).nunique()),
    }


def resolve_pipeline_inputs(
    eyetrack_path=None,
    records_path=None,
    subject_dir=None,
    task_name="T6_EmotionalObservationTask",
    cleaned_output_dir=None,
):
    cleaned_output_dir = (
        Path(cleaned_output_dir).expanduser().resolve()
        if cleaned_output_dir is not None
        else None
    )
    if subject_dir is not None:
        subject_dir = Path(subject_dir).expanduser().resolve()
        cleaned = prepare_clean_raw_data(
            subject_dir=subject_dir,
            task_name=task_name,
            output_dir=cleaned_output_dir,
        )
        return {
            "eyetrack_path": Path(cleaned["formal_eyetrack_path"]),
            "records_path": Path(cleaned["formal_records_path"]),
            "subject_dir": subject_dir,
            "task_name": task_name,
            "selected_run_dir": cleaned["selected_run_dir"],
            "input_mode": "auto_cleaned_subject_dir",
            "subject_id": subject_dir.name,
        }

    if eyetrack_path is None and records_path is None:
        raise ValueError("必须提供 eyetrack_path 和 records_path，或提供 subject_dir。")
    if eyetrack_path is None or records_path is None:
        raise ValueError("eyetrack_path 和 records_path 必须同时提供。")

    eyetrack_path = Path(eyetrack_path).expanduser().resolve()
    records_path = Path(records_path).expanduser().resolve()
    if not eyetrack_path.is_file():
        raise FileNotFoundError(f"眼动文件不存在: {eyetrack_path}")
    if not records_path.is_file():
        raise FileNotFoundError(f"记录文件不存在: {records_path}")

    inferred_subject_dir = eyetrack_path.parent
    if inferred_subject_dir.name == "raw_data":
        selected_run_dir = str(inferred_subject_dir.parent)
        inferred_subject_dir = inferred_subject_dir.parents[2]
    else:
        selected_run_dir = ""

    return {
        "eyetrack_path": eyetrack_path,
        "records_path": records_path,
        "subject_dir": inferred_subject_dir,
        "task_name": task_name,
        "selected_run_dir": selected_run_dir,
        "input_mode": "direct_files",
        "subject_id": inferred_subject_dir.name,
    }


def run_pipeline(
    eyetrack_path=None,
    records_path=None,
    subject_dir=None,
    task_name="T6_EmotionalObservationTask",
    cleaned_output_dir=None,
    init_params=None,
    bounds=None,
    timestamp_unit_scale=1e6,
):
    resolved_inputs = resolve_pipeline_inputs(
        eyetrack_path=eyetrack_path,
        records_path=records_path,
        subject_dir=subject_dir,
        task_name=task_name,
        cleaned_output_dir=cleaned_output_dir,
    )

    eyetrack_path = resolved_inputs["eyetrack_path"]
    records_path = resolved_inputs["records_path"]
    eyetrack_df = pd.read_csv(eyetrack_path, encoding="utf-8-sig")
    records_df = pd.read_csv(records_path, encoding="utf-8-sig")

    base_df, detected_trials = build_base_frame(eyetrack_df)
    annotated_df = annotate_spatial_and_emotion(base_df, records_df)
    prepared_df, model_data = prepare_model_data(
        annotated_df, timestamp_unit_scale=timestamp_unit_scale
    )
    optimization_result = optimize_free_parameters(
        model_data,
        init_params=init_params,
        bounds=bounds,
    )

    best_params = optimization_result["best_params"]
    return {
        "input_mode": resolved_inputs["input_mode"],
        "subject_id": resolved_inputs["subject_id"],
        "task_name": resolved_inputs["task_name"],
        "selected_run_dir": resolved_inputs["selected_run_dir"],
        "Pm1_positive": optimization_result["Pm1_positive"],
        "Pm2_negative": optimization_result["Pm2_negative"],
        "Pm3_neutral": optimization_result["Pm3_neutral"],
        "Tmax1": best_params["Tmax1"],
        "Tmax2": best_params["Tmax2"],
        "Tmax3": best_params["Tmax3"],
        "dc": float(px_to_visual_angle_deg(best_params["dc"])),
    }


def main():
    core_dir = Path(__file__).resolve().parent
    sample_subject_dir = core_dir / "data_example" / "TEST001_demo_female"
    task_name = "T6_EmotionalObservationTask"
    output_dir = core_dir / "result_example"

    parts = sample_subject_dir.name.split("_")
    name = parts[1]
    gender = parts[2]

    result = run_pipeline(
        subject_dir=sample_subject_dir,
        task_name=task_name,
    )

    thresholds = load_thresholds(ROOT_DIR / "devide_rule")
    category = classify(
        result["Pm2_negative"],
        result["Tmax2"],
        float(thresholds["pm2_cutoff_q75"]),
        float(thresholds["tmax2_cutoff_median"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = output_dir / f"{name}_{timestamp}.csv"
    row = {
        "name": name,
        "gender": gender,
        "Pm1_positive": result["Pm1_positive"],
        "Pm2_negative": result["Pm2_negative"],
        "Pm3_neutral": result["Pm3_neutral"],
        "Tmax1": result["Tmax1"],
        "Tmax2": result["Tmax2"],
        "Tmax3": result["Tmax3"],
        "dc": result["dc"],
        "类别": category,
    }
    pd.DataFrame([row]).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(output_file)


if __name__ == "__main__":
    main()
