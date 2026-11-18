import numpy as np
import pandas as pd


def build_a1_df(input_csv_path):
    return build_a1_df_from_df(pd.read_csv(input_csv_path))


def build_a1_df_from_df(eyetrack_df):
    """
    从 formal_eyetrack.csv 构建 A1。

    这一层只保留眼动原始时间戳、坐标，以及根据 trigger 粗略推断出的 trial_number。
    """

    df = eyetrack_df.copy()

    required_cols = [
        "timestamp",
        "bino_eye_gaze_position_x",
        "bino_eye_gaze_position_y",
        "bino_eye_valid",
        "trigger",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")

    # trigger >= 100 代表已经进入正式试次编码区间。
    trial_raw = np.where(df["trigger"] >= 100, df["trigger"] // 100, np.nan)
    df["trial_number"] = pd.Series(trial_raw).ffill()
    df["trial_number"] = df["trial_number"].fillna(0).astype(int)

    a1 = df[
        [
            "timestamp",
            "bino_eye_gaze_position_x",
            "bino_eye_gaze_position_y",
            "bino_eye_valid",
            "trial_number",
        ]
    ].copy()

    a1.rename(
        columns={
            "bino_eye_gaze_position_x": "gaze_x",
            "bino_eye_gaze_position_y": "gaze_y",
            "bino_eye_valid": "gaze_valid",
        },
        inplace=True,
    )
    return a1


def build_a2_df(a1_df, eyetrack_csv_path):
    return build_a2_df_from_df(a1_df, pd.read_csv(eyetrack_csv_path))


def build_a2_df_from_df(a1_df, eyetrack_df):
    """
    基于 A1 和 trigger 构建 A2。

    A2 主要补出每一行当前处于 fixation / picture / blank 哪个阶段。
    """

    a1 = a1_df.copy()
    raw = eyetrack_df.copy()

    if "trigger" not in raw.columns:
        raise ValueError("formal_eyetrack.csv 中缺少 trigger 列")

    if len(a1) != len(raw):
        raise ValueError(
            f"A1 行数({len(a1)}) 与 formal_eyetrack.csv 行数({len(raw)}) 不一致，无法逐行对应。"
        )

    trigger = pd.to_numeric(raw["trigger"], errors="coerce").fillna(0).astype(int)

    fixation = []
    picture = []
    blank = []
    current_stage = None

    for value in trigger:
        # 编码尾数约定：
        # 00 = fixation
        # 01 = picture
        # 02 = blank
        mod = value % 100
        if value >= 100:
            if mod == 0:
                current_stage = "fixation"
            elif mod == 1:
                current_stage = "picture"
            elif mod == 2:
                current_stage = "blank"

        fixation.append(1 if current_stage == "fixation" else 0)
        picture.append(1 if current_stage == "picture" else 0)
        blank.append(1 if current_stage == "blank" else 0)

    a2 = a1.copy()
    a2["fixation"] = fixation
    a2["picture"] = picture
    a2["blank"] = blank
    return a2


def build_a3_df(a2_df, records_csv_path):
    return build_a3_df_from_df(a2_df, pd.read_csv(records_csv_path))


def build_a3_df_from_df(a2_df, records_df):
    """
    基于 A2 和 formal_records.csv 构建 A3。

    A3 的核心作用是把 picture 阶段的每一行映射到具体情绪标签。
    """

    a2 = a2_df.copy()
    records = records_df.copy()

    required_a2_cols = ["trial_number", "picture"]
    required_records_cols = ["试次轮次", "刺激图片情绪类型"]

    missing_a2 = [col for col in required_a2_cols if col not in a2.columns]
    missing_records = [col for col in required_records_cols if col not in records.columns]

    if missing_a2:
        raise ValueError(f"A2 缺少必要列: {missing_a2}")
    if missing_records:
        raise ValueError(f"formal_records.csv 缺少必要列: {missing_records}")

    records = records.copy()
    records["刺激图片情绪类型"] = (
        records["刺激图片情绪类型"].astype(str).str.strip().str.lower()
    )

    # nagetive 是历史数据里的兼容拼写，这里统一映射为 negative。
    emotion_map = {
        "positive": "positive",
        "negative": "negative",
        "nagetive": "negative",
        "neutral": "neutral",
        "complex": "complex",
    }
    records["emotion_std"] = records["刺激图片情绪类型"].map(emotion_map)

    unknown_emotions = records.loc[
        records["emotion_std"].isna(), "刺激图片情绪类型"
    ].unique()
    if len(unknown_emotions) > 0:
        raise ValueError(
            f"formal_records.csv 中存在无法识别的情绪类型: {list(unknown_emotions)}"
        )

    trial_to_emotion = dict(zip(records["试次轮次"], records["emotion_std"]))

    a3 = a2.copy()
    for emotion in ["positive", "negative", "neutral", "complex"]:
        a3[emotion] = 0

    current_emotion = a3["trial_number"].map(trial_to_emotion)
    pic_mask = a3["picture"] == 1

    a3.loc[pic_mask & (current_emotion == "positive"), "positive"] = 1
    a3.loc[pic_mask & (current_emotion == "negative"), "negative"] = 1
    a3.loc[pic_mask & (current_emotion == "neutral"), "neutral"] = 1
    a3.loc[pic_mask & (current_emotion == "complex"), "complex"] = 1
    return a3


def build_a4_df(a3_df):
    """
    基于 A3 构建 A4。

    A4 会在 picture 阶段按实验显示区域过滤 gaze 点，去掉明显不在图片区域内的采样。
    """

    df = a3_df.copy()

    required_cols = ["timestamp", "gaze_x", "gaze_y", "trial_number", "picture"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"A3 缺少必要列: {missing_cols}")

    # 事件时间由完整 trigger 映射确定，不能随 gaze 有效性或图片坐标过滤而移动。
    for col in ["timestamp", "trial_number", "picture"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trial_start_time"] = df.groupby("trial_number")["timestamp"].transform("min")
    picture_start_map = df[df["picture"] == 1].groupby("trial_number")["timestamp"].min()
    df["picture_start_time"] = df["trial_number"].map(picture_start_map)

    # 这里使用当前实验的固定图片显示区域。
    x_min, x_max = 480, 1440
    y_min, y_max = 180, 900

    in_picture_area = (
        (df["gaze_x"] >= x_min)
        & (df["gaze_x"] <= x_max)
        & (df["gaze_y"] >= y_min)
        & (df["gaze_y"] <= y_max)
    )
    keep_mask = (df["picture"] != 1) | ((df["picture"] == 1) & in_picture_area)

    a4 = df.loc[keep_mask].copy()
    a4.reset_index(drop=True, inplace=True)
    return a4


def build_a5_df(a4_df):
    """
    基于 A4 构建 A5。

    A5 是模型拟合真正消费的输入表，重点是补出相对时间字段：
    - t_from_trial_start
    - t_from_picture_start
    """

    df = a4_df.copy()

    required_cols = [
        "timestamp",
        "gaze_x",
        "gaze_y",
        "gaze_valid",
        "trial_number",
        "trial_start_time",
        "picture_start_time",
        "fixation",
        "picture",
        "positive",
        "negative",
        "neutral",
        "complex",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"A4 缺少必要列: {missing_cols}")

    numeric_cols = list(required_cols)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 设备标记为无效的采样点不进入模型；其余坐标规则保持不变。
    df = df[
        df["gaze_valid"].eq(1) & (df["gaze_x"] >= 0) & (df["gaze_y"] >= 0)
    ].copy()

    df["t_from_trial_start"] = df["timestamp"] - df["trial_start_time"]
    df["t_from_picture_start"] = df["timestamp"] - df["picture_start_time"]

    # 模型当前只使用 fixation 和 picture 阶段。
    df = df[(df["fixation"] == 1) | (df["picture"] == 1)].copy()

    output_cols = [
        "gaze_x",
        "gaze_y",
        "trial_number",
        "positive",
        "negative",
        "neutral",
        "complex",
        "t_from_trial_start",
        "t_from_picture_start",
    ]
    a5 = df[output_cols].copy()
    a5.reset_index(drop=True, inplace=True)
    return a5


def run_memory_pipeline(eyetrack_csv_path, records_csv_path):
    """
    执行 T5 的完整内存流水线。

    输入：
    - eyetrack_csv_path: raw_data/formal_eyetrack.csv
    - records_csv_path: raw_data/formal_records.csv

    输出：
    - 返回 A1 到 A5 五个阶段的 DataFrame，全部只在内存中传递
    """

    eyetrack_df = pd.read_csv(eyetrack_csv_path)
    records_df = pd.read_csv(records_csv_path)
    return run_memory_pipeline_from_dataframes(eyetrack_df, records_df)


def run_memory_pipeline_from_dataframes(eyetrack_df, records_df):
    a1 = build_a1_df_from_df(eyetrack_df)
    a2 = build_a2_df_from_df(a1, eyetrack_df)
    a3 = build_a3_df_from_df(a2, records_df)
    a4 = build_a4_df(a3)
    a5 = build_a5_df(a4)

    return {
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "A4": a4,
        "A5": a5,
    }
