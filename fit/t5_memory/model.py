import time

import numpy as np
from scipy.optimize import minimize


def _build_grid(width, height, m, n):
    xs = (np.arange(m) + 0.5) * (width / m)
    ys = (np.arange(n) + 0.5) * (height / n)
    x_center, y_center = np.meshgrid(xs, ys, indexing="ij")
    return x_center, y_center


def _gaussian_density_on_grid(x, y, x_center, y_center, d_c):
    dist2 = (x_center - x) ** 2 + (y_center - y) ** 2
    return np.exp(-dist2 / (2.0 * d_c**2)) / (2.0 * np.pi * d_c**2)


def _spatial_kernel_on_grid(x, y, x_center, y_center, sigma_s):
    dist2 = (x_center - x) ** 2 + (y_center - y) ** 2
    return np.exp(-dist2 / (2.0 * sigma_s**2))


def _vector_to_param_dict(theta):
    names = ["positive", "negative", "neutral", "complex"]
    out = {}
    index = 0
    for emotion in names:
        out[emotion] = {
            "tau": float(theta[index]),
            "sigma_s": float(theta[index + 1]),
            "d_c": float(theta[index + 2]),
        }
        index += 3
    return out


def load_a5_dataframe(
    df,
    timestamp_unit_scale=1e6,
    x_min=480,
    y_min=180,
    width=960,
    height=720,
    resample_ratio=1.0,
):
    """
    从 A5 的 DataFrame 读取数据，并整理为模型所需内存结构。
    """

    if not (0 < resample_ratio <= 1):
        raise ValueError("resample_ratio 必须满足 0 < resample_ratio <= 1")

    df = df.copy()

    if "negative" not in df.columns and "nagetive" in df.columns:
        df["negative"] = df["nagetive"]

    required_cols = [
        "gaze_x",
        "gaze_y",
        "trial_number",
        "t_from_picture_start",
        "positive",
        "negative",
        "neutral",
        "complex",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"A5 缺少必要列: {missing}")

    emotion_cols = ["positive", "negative", "neutral", "complex"]

    df["emotion_sum"] = df[emotion_cols].sum(axis=1)
    pic_df = df[df["emotion_sum"] == 1].copy()

    if len(pic_df) == 0:
        raise ValueError("A5 中没有识别到图片阶段数据。")

    num_cols = ["gaze_x", "gaze_y", "trial_number", "t_from_picture_start"] + emotion_cols
    for col in num_cols:
        pic_df[col] = np.asarray(pic_df[col], dtype=float)

    pic_df = pic_df.dropna(
        subset=["gaze_x", "gaze_y", "trial_number", "t_from_picture_start"]
    ).copy()

    pic_df["x_local"] = pic_df["gaze_x"] - x_min
    pic_df["y_local"] = pic_df["gaze_y"] - y_min
    pic_df["t_sec"] = pic_df["t_from_picture_start"] / float(timestamp_unit_scale)

    trials = []
    for trial_number, group in pic_df.groupby("trial_number", sort=True):
        group = group.sort_values("t_sec").copy()

        emotion_name = None
        for emo in emotion_cols:
            if group[emo].sum() > 0:
                emotion_name = emo
                break

        if emotion_name is None:
            continue

        n_total = len(group)
        if resample_ratio >= 1.0 or n_total <= 1:
            group_sub = group
        else:
            n_keep = max(1, int(np.floor(n_total * resample_ratio)))
            keep_idx = np.linspace(0, n_total - 1, n_keep)
            keep_idx = np.unique(np.round(keep_idx).astype(int))

            if len(keep_idx) < n_keep:
                extra = np.setdiff1d(np.arange(n_total), keep_idx)
                need = n_keep - len(keep_idx)
                keep_idx = np.sort(np.concatenate([keep_idx, extra[:need]]))

            group_sub = group.iloc[keep_idx].copy()

        trials.append(
            {
                "trial_number": int(trial_number),
                "emotion": emotion_name,
                "x": group_sub["x_local"].to_numpy(dtype=float),
                "y": group_sub["y_local"].to_numpy(dtype=float),
                "t": group_sub["t_sec"].to_numpy(dtype=float),
                "n": len(group_sub),
                "n_original": n_total,
            }
        )

    if len(trials) == 0:
        raise ValueError("没有成功整理出任何图片阶段 trial。")

    return {
        "width": float(width),
        "height": float(height),
        "emotions": emotion_cols,
        "resample_ratio": float(resample_ratio),
        "trials": trials,
    }


def loglik_picture_only(data_dict, theta, M=12, N=9, eps=1e-300, return_details=False):
    """
    根据自由参数 theta，计算所有图片阶段观测点的 log p(x,y | t_k) 之和。
    """

    theta = np.asarray(theta, dtype=float)
    if theta.shape[0] != 12:
        raise ValueError("theta 必须是长度为 12 的向量。")

    param_dict = _vector_to_param_dict(theta)
    width = data_dict["width"]
    height = data_dict["height"]
    trials = data_dict["trials"]

    x_center, y_center = _build_grid(width, height, M, N)

    total_loglik = 0.0
    details = []

    for trial in trials:
        emotion = trial["emotion"]
        x_arr = trial["x"]
        y_arr = trial["y"]
        t_arr = trial["t"]

        tau = param_dict[emotion]["tau"]
        sigma_s = param_dict[emotion]["sigma_s"]
        d_c = param_dict[emotion]["d_c"]

        attention = np.ones((M, N), dtype=float)
        last_t = None
        for index, current_t in enumerate(t_arr):
            x_value = x_arr[index]
            y_value = y_arr[index]

            if last_t is not None:
                dt = max(0.0, current_t - last_t)
                alpha = np.exp(-dt / tau)
                attention *= alpha

            probability = attention / max(attention.sum(), eps)
            entropy = -np.sum(probability * np.log(np.maximum(probability, eps)))

            gaussian = _gaussian_density_on_grid(x_value, y_value, x_center, y_center, d_c)
            p_xy = float(np.sum(probability * gaussian))
            p_xy = max(p_xy, eps)

            logp = np.log(p_xy)
            total_loglik += logp

            if return_details:
                details.append(
                    {
                        "trial_number": trial["trial_number"],
                        "emotion": emotion,
                        "k": index,
                        "t_sec": current_t,
                        "x": x_value,
                        "y": y_value,
                        "p": p_xy,
                        "logp": logp,
                        "P": probability.copy(),
                        "H_s": float(entropy),
                    }
                )

            kernel = _spatial_kernel_on_grid(x_value, y_value, x_center, y_center, sigma_s)
            attention += kernel
            last_t = current_t

    if return_details:
        return total_loglik, details
    return total_loglik


def compute_interpretable_metrics(data_dict, theta, M=12, N=9, r_focus=50.0, eps=1e-300):
    """
    计算时间半衰期、归一化平均空间熵、单位聚焦点信息量。
    """

    theta = np.asarray(theta, dtype=float)
    param_dict = _vector_to_param_dict(theta)

    _, details = loglik_picture_only(
        data_dict=data_dict,
        theta=theta,
        M=M,
        N=N,
        eps=eps,
        return_details=True,
    )

    log_mn = np.log(M * N)
    entropy_by_emotion = {
        "positive": [],
        "negative": [],
        "neutral": [],
        "complex": [],
    }

    for item in details:
        entropy_by_emotion[item["emotion"]].append(item["H_s"])

    metrics = {}
    for emotion, params in param_dict.items():
        tau = params["tau"]
        d_c = params["d_c"]
        h_values = entropy_by_emotion[emotion]

        if h_values:
            h_bar_norm = float(np.mean(np.array(h_values) / log_mn))
        else:
            h_bar_norm = np.nan

        metrics[emotion] = {
            "T_half": float(tau * np.log(2.0)),
            "H_bar_norm": h_bar_norm,
            "I_focus": float(1.0 - np.exp(-(r_focus**2) / (2.0 * d_c**2))),
            "tau": float(tau),
            "d_c": float(d_c),
            "r_focus": float(r_focus),
        }

    return metrics


def fit_free_parameters(
    data_dict,
    M=12,
    N=9,
    init_theta=None,
    bounds=None,
    method="L-BFGS-B",
    r_focus=50.0,
    progress_callback=None,
):
    """
    最大化图片阶段 log-likelihood，拟合 12 个自由参数。
    """

    if init_theta is None:
        init_theta = np.array(
            [
                0.26,
                27.0,
                31.0,
                0.18,
                38.0,
                32.0,
                0.14,
                34.0,
                34.0,
                0.15,
                31.0,
                28.0,
            ],
            dtype=float,
        )

    if bounds is None:
        bounds = [
            (0.01, 10.0),
            (5.0, 600.0),
            (5.0, 400.0),
            (0.01, 10.0),
            (5.0, 600.0),
            (5.0, 400.0),
            (0.01, 10.0),
            (5.0, 600.0),
            (5.0, 400.0),
            (0.0001, 10.0),
            (5.0, 600.0),
            (5.0, 400.0),
        ]

    init_theta = np.asarray(init_theta, dtype=float)
    if init_theta.shape[0] != 12:
        raise ValueError("init_theta 必须是长度为 12 的向量。")
    if len(bounds) != 12:
        raise ValueError("bounds 必须是长度为 12 的列表。")

    def objective(theta):
        return -loglik_picture_only(
            data_dict=data_dict,
            theta=theta,
            M=M,
            N=N,
            return_details=False,
        )

    fit_started_ts = time.time()
    iter_count = 0

    if progress_callback is not None:
        progress_callback({"stage": "fit_model", "status": "start"})

    def callback(theta):
        nonlocal iter_count
        iter_count += 1
        if progress_callback is None:
            return
        negloglik = float(objective(theta))
        progress_callback(
            {
                "stage": "fit_model",
                "status": "iter",
                "iteration": iter_count,
                "elapsed_seconds": round(time.time() - fit_started_ts, 3),
                "negloglik": negloglik,
            }
        )

    opt = minimize(
        objective,
        x0=init_theta,
        method=method,
        bounds=bounds,
        callback=callback,
    )

    theta_opt = opt.x
    best_loglik = -opt.fun
    param_dict = _vector_to_param_dict(theta_opt)

    interpretable_metrics = compute_interpretable_metrics(
        data_dict=data_dict,
        theta=theta_opt,
        M=M,
        N=N,
        r_focus=r_focus,
    )

    return {
        "theta_opt": theta_opt,
        "param_dict": param_dict,
        "interpretable_metrics": interpretable_metrics,
        "best_loglik": best_loglik,
        "success": bool(opt.success),
        "message": opt.message,
        "optimizer_result": opt,
    }


def flatten_fit_result(fit_res):
    """
    将拟合结果展开为适合写入 CSV 的扁平字典。
    """

    row = {
        "success": bool(fit_res["success"]),
        "message": str(fit_res["message"]),
        "best_loglik": float(fit_res["best_loglik"]),
    }

    emotions = ["positive", "negative", "neutral", "complex"]
    for emotion in emotions:
        params = fit_res["param_dict"][emotion]
        metrics = fit_res["interpretable_metrics"][emotion]

        row[f"{emotion}_tau"] = float(params["tau"])
        row[f"{emotion}_sigma_s"] = float(params["sigma_s"])
        row[f"{emotion}_d_c"] = float(params["d_c"])
        row[f"{emotion}_T_half"] = float(metrics["T_half"])
        row[f"{emotion}_H_bar_norm"] = float(metrics["H_bar_norm"])
        row[f"{emotion}_I_focus"] = float(metrics["I_focus"])

    return row


def run_model_for_a5_df(
    a5_df,
    timestamp_unit_scale=1e9,
    x_min=480,
    y_min=180,
    width=960,
    height=720,
    resample_ratio=0.1,
    M=12,
    N=9,
    r_focus=50.0,
    init_theta=None,
    progress_callback=None,
):
    """
    基于 A5 的 DataFrame 执行完整拟合流程。
    """

    if init_theta is None:
        init_theta = np.array(
            [
                0.50,
                120.0,
                80.0,
                0.50,
                120.0,
                80.0,
                0.50,
                120.0,
                80.0,
                0.50,
                120.0,
                80.0,
            ],
            dtype=float,
        )
    else:
        init_theta = np.asarray(init_theta, dtype=float)

    data = load_a5_dataframe(
        df=a5_df,
        timestamp_unit_scale=timestamp_unit_scale,
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height,
        resample_ratio=resample_ratio,
    )

    initial_loglik = loglik_picture_only(data, init_theta, M=M, N=N)

    fit_res = fit_free_parameters(
        data_dict=data,
        M=M,
        N=N,
        init_theta=init_theta,
        r_focus=r_focus,
        progress_callback=progress_callback,
    )
    fit_res["initial_loglik"] = float(initial_loglik)
    fit_res["flat_result"] = flatten_fit_result(fit_res)
    fit_res["flat_result"]["initial_loglik"] = float(initial_loglik)
    fit_res["flat_result"]["a5_rows"] = int(len(a5_df))
    return fit_res
