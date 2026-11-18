from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


RAW_DIRNAME = "raw_data"
SEGMENT_RE = re.compile(r"^formal_(\d+)_eyetrack\.csv$")


def load_clean_t5_raw_data(task_run_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_run_dir = Path(task_run_dir)
    raw_data_dir = task_run_dir / RAW_DIRNAME
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"缺少 raw_data 目录: {raw_data_dir}")
    return load_eyetrack_df(raw_data_dir), load_records_df(raw_data_dir)


def load_records_df(raw_data_dir) -> pd.DataFrame:
    raw_data_dir = Path(raw_data_dir)
    formal_records = raw_data_dir / "formal_records.csv"
    formal_recording = raw_data_dir / "formal_recording.csv"
    if formal_records.is_file():
        return pd.read_csv(formal_records)
    if formal_recording.is_file():
        return pd.read_csv(formal_recording)
    raise FileNotFoundError(
        f"缺少 formal_records.csv 或 formal_recording.csv: {raw_data_dir}"
    )


def load_eyetrack_df(raw_data_dir) -> pd.DataFrame:
    raw_data_dir = Path(raw_data_dir)
    single_path = raw_data_dir / "formal_eyetrack.csv"
    segment_paths = []
    seen_numbers = set()

    for path in raw_data_dir.iterdir():
        if not path.is_file():
            continue
        match = SEGMENT_RE.match(path.name)
        if not match:
            continue
        number = int(match.group(1))
        if number in seen_numbers:
            raise ValueError(f"分段眼动编号重复: {path.name}")
        seen_numbers.add(number)
        segment_paths.append((number, path))

    if single_path.is_file() and segment_paths:
        raise ValueError("同时存在 formal_eyetrack.csv 和分段 formal_*_eyetrack.csv")
    if single_path.is_file():
        return pd.read_csv(single_path)
    if segment_paths:
        ordered_paths = [path for _, path in sorted(segment_paths, key=lambda item: item[0])]
        return merge_segmented_eyetrack_dfs(ordered_paths)
    raise FileNotFoundError(
        f"缺少 formal_eyetrack.csv 或分段 formal_*_eyetrack.csv: {raw_data_dir}"
    )


def merge_segmented_eyetrack_dfs(segment_paths) -> pd.DataFrame:
    frames = []
    first_columns = None
    for segment_path in segment_paths:
        df = pd.read_csv(segment_path)
        if first_columns is None:
            first_columns = list(df.columns)
        elif list(df.columns) != first_columns:
            raise ValueError(f"分段眼动列结构不一致: {Path(segment_path).name}")
        frames.append(df)

    if not frames:
        raise ValueError("没有可拼接的分段眼动文件。")
    return pd.concat(frames, ignore_index=True)
