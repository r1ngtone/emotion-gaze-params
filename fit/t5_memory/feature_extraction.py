import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from single_subject.clean_input import load_clean_t5_raw_data
from t5_memory.model import run_model_for_a5_df
from t5_memory.pipeline import run_memory_pipeline_from_dataframes

TASK_DIRNAME = "T5_EmotionalOrientingTask"
RAW_DIRNAME = "raw_data"
FREE_PARAMETER_COLUMNS = [
    "positive_tau",
    "positive_sigma_s",
    "positive_d_c",
    "negative_tau",
    "negative_sigma_s",
    "negative_d_c",
    "neutral_tau",
    "neutral_sigma_s",
    "neutral_d_c",
    "complex_tau",
    "complex_sigma_s",
    "complex_d_c",
]
INTERPRETABLE_METRIC_COLUMNS = [
    "positive_T_half",
    "positive_H_bar_norm",
    "positive_I_focus",
    "negative_T_half",
    "negative_H_bar_norm",
    "negative_I_focus",
    "neutral_T_half",
    "neutral_H_bar_norm",
    "neutral_I_focus",
    "complex_T_half",
    "complex_H_bar_norm",
    "complex_I_focus",
]


def _utcnow_text():
    return datetime.now().isoformat(timespec="seconds")


def _parse_subject_meta(subject_dir: Path) -> dict:
    parts = subject_dir.name.split("_")
    return {
        "subject_id": parts[0] if len(parts) > 0 else subject_dir.name,
        "subject_name": parts[1] if len(parts) > 1 else "",
        "gender": parts[2] if len(parts) > 2 else "",
    }


def _emit_progress(progress_callback, payload):
    if progress_callback is not None:
        progress_callback(payload)


def _extract_subject_features_core(
    subject_meta,
    eyetrack_df,
    records_df,
    resample_ratio=0.1,
    progress_callback=None,
) -> dict:
    started_at = _utcnow_text()
    started_ts = time.time()

    _emit_progress(
        progress_callback,
        {
            "stage": "load_clean_input",
            "status": "done",
            "eyetrack_rows": int(len(eyetrack_df)),
            "records_rows": int(len(records_df)),
        },
    )

    # 第一步：把原始眼动数据整理成模型真正需要的 A5 表。
    # 这一层还是沿用你已有的底层算法，不改研究定义。
    pipeline_data = run_memory_pipeline_from_dataframes(
        eyetrack_df=eyetrack_df,
        records_df=records_df,
    )
    _emit_progress(
        progress_callback,
        {
            "stage": "build_a5",
            "status": "done",
            "a5_rows": int(len(pipeline_data["A5"])),
        },
    )
    # 第二步：直接用 A5 跑自由参数拟合和解释性指标计算。
    # 这里只拿算法结果，不让它额外写 fit_results.csv 等中间文件。
    fit_result = run_model_for_a5_df(
        a5_df=pipeline_data["A5"],
        timestamp_unit_scale=1e9,
        x_min=480,
        y_min=180,
        width=960,
        height=720,
        resample_ratio=resample_ratio,
        M=12,
        N=9,
        r_focus=50.0,
        progress_callback=progress_callback,
    )
    flat_result = dict(fit_result["flat_result"])
    _emit_progress(
        progress_callback,
        {
            "stage": "fit_model",
            "status": "done",
            "best_loglik": float(fit_result["best_loglik"]),
            "elapsed_seconds": round(time.time() - started_ts, 3),
        },
    )

    # 这里把 12 个自由参数单独拆出来。
    # 这样最终 CSV 里既能保留模型参数，也方便你后续单独查看。
    free_parameters = {
        column: flat_result[column]
        for column in FREE_PARAMETER_COLUMNS
    }
    # 这里把 12 个解释性指标单独拆出来。
    # 这 12 个指标既会写入最终 CSV，也会作为后续分类的输入。
    interpretable_metrics = {
        column: flat_result[column]
        for column in INTERPRETABLE_METRIC_COLUMNS
    }

    return {
        "subject_meta": dict(subject_meta),
        "fit_meta": {
            "status": "success",
            "error_message": "",
            "started_at": started_at,
            "finished_at": _utcnow_text(),
            "subject_elapsed_seconds": round(time.time() - started_ts, 3),
            "success": bool(flat_result["success"]),
            "message": str(flat_result["message"]),
            "best_loglik": float(flat_result["best_loglik"]),
            "initial_loglik": float(flat_result["initial_loglik"]),
            "a5_rows": int(flat_result["a5_rows"]),
        },
        "free_parameters": free_parameters,
        "interpretable_metrics": interpretable_metrics,
        "classification_features": dict(interpretable_metrics),
        "raw_result_row": flat_result,
    }


def extract_subject_features(task_run_dir, resample_ratio=0.1, progress_callback=None) -> dict:
    """
    这里负责单被试参数提取，但故意不写任何结果文件。

    原因很直接：
    你要求单被试最终只能有一个 CSV。
    所以这里不走会自动落盘的高层封装，
    而是只复用底层算法，把结果保留在内存里，交给后面的分类和单 CSV 输出。
    """

    task_run_dir = Path(task_run_dir)
    raw_data_dir = task_run_dir / RAW_DIRNAME
    subject_dir = task_run_dir.parent.parent
    eyetrack_df, records_df = load_clean_t5_raw_data(task_run_dir)
    subject_meta = {
        **_parse_subject_meta(subject_dir),
        "task_run_dir": str(task_run_dir),
        "raw_data_dir": str(raw_data_dir),
        "task_type": "T5",
        "task_name": TASK_DIRNAME,
    }
    return _extract_subject_features_core(
        subject_meta=subject_meta,
        eyetrack_df=eyetrack_df,
        records_df=records_df,
        resample_ratio=resample_ratio,
        progress_callback=progress_callback,
    )


def extract_subject_features_from_flat_dir(
    subject_dir,
    resample_ratio=0.1,
    progress_callback=None,
) -> dict:
    subject_dir = Path(subject_dir)
    eyetrack_csv = subject_dir / "formal_eyetrack.csv"
    records_csv = subject_dir / "formal_records.csv"

    if not eyetrack_csv.is_file():
        raise FileNotFoundError(f"缺少文件: {eyetrack_csv}")
    if not records_csv.is_file():
        raise FileNotFoundError(f"缺少文件: {records_csv}")

    eyetrack_df = pd.read_csv(eyetrack_csv)
    records_df = pd.read_csv(records_csv)
    subject_meta = {
        **_parse_subject_meta(subject_dir),
        "task_run_dir": "",
        "raw_data_dir": str(subject_dir),
        "task_type": "T5",
        "task_name": TASK_DIRNAME,
    }
    return _extract_subject_features_core(
        subject_meta=subject_meta,
        eyetrack_df=eyetrack_df,
        records_df=records_df,
        resample_ratio=resample_ratio,
        progress_callback=progress_callback,
    )
