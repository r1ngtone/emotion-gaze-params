"""
合成数据演示脚本：生成不含真实被试的模拟注视轨迹，并演示参数拟合与报告量计算。

用途（面向 BRM 复现）：
    - 在真实数据因参与者隐私无法公开的情况下，用合成数据演示"拟合入口 + 报告量计算入口"
      可以独立运行，并给出预期输出，供读者核对。
    - 合成数据由一组已知的"真值参数"生成；随后用同样的拟合流程恢复这些参数，
      以证明参数估计流程的可复现性（参数恢复 / parameter recovery 的最小示例）。

运行方式（在本仓库根目录下）：
    python example/generate_synthetic_demo.py

依赖：
    numpy, scipy, pandas

说明：
    本脚本直接生成模型消费的 A5 结构（见 fit/t5_memory/pipeline.py 的 build_a5_df），
    从而绕开采集阶段的 trigger 细节；合成数据不包含任何真实被试记录。
"""

import os
import sys

import numpy as np
import pandas as pd

# 允许直接 import fit/ 下的模块（本脚本位于 example/ 子目录）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fit.t5_memory.model import (
    fit_free_parameters,
    compute_interpretable_metrics,
    load_a5_dataframe,
)

# ---------------------------------------------------------------------------
# 屏幕坐标与采样配置（与论文方法一致）
# ---------------------------------------------------------------------------
X_MIN, Y_MIN = 480, 180
WIDTH, HEIGHT = 960, 720
TIMESTAMP_UNIT_SCALE = 1e6          # 微秒 -> 秒
SAMPLING_RATE_HZ = 200.0            # 眼动仪采样率

# 四类情绪，顺序与 _vector_to_param_dict 一致
EMOTIONS = ["positive", "negative", "neutral", "complex"]

# ---------------------------------------------------------------------------
# 已知真值参数（theta 为 12 维：每类情绪 tau, sigma_s, d_c）
# ---------------------------------------------------------------------------
TRUE_THETA = np.array(
    [
        # positive: tau, sigma_s, d_c
        0.26, 27.0, 31.0,
        # negative: tau, sigma_s, d_c
        0.18, 38.0, 32.0,
        # neutral: tau, sigma_s, d_c
        0.14, 34.0, 34.0,
        # complex: tau, sigma_s, d_c
        0.15, 31.0, 28.0,
    ],
    dtype=float,
)

# 每类情绪生成的试次数（演示用，取小规模以保证秒级跑完）
TRIALS_PER_EMOTION = 2
# 每个试次图片阶段的时长（秒）
PICTURE_DURATION_S = 5.0


def _sample_attention_sequence(
    tau, sigma_s, d_c, duration_s, sampling_rate, seed
):
    """按模型的注意力机制，采样一个试次内的注视轨迹 (x, y, t)。

    采用与 model.loglik_picture_only 相同的动力学：注意力在 12x9 网格上
    累积，随 dt/tau 指数衰减；每个采样点的注视位置从"注意力加权的高斯"
    中抽取。这样生成的轨迹在统计上受 (tau, sigma_s, d_c) 支配。
    """
    rng = np.random.default_rng(seed)

    M, N = 12, 9
    xs = (np.arange(M) + 0.5) * (WIDTH / M)
    ys = (np.arange(N) + 0.5) * (HEIGHT / N)
    x_center, y_center = np.meshgrid(xs, ys, indexing="ij")

    n_steps = int(duration_s * sampling_rate)
    t_arr = np.arange(n_steps) / sampling_rate

    attention = np.ones((M, N), dtype=float)
    xs_out, ys_out = [], []

    last_t = None
    for k, t in enumerate(t_arr):
        if last_t is not None:
            dt = max(0.0, t - last_t)
            attention *= np.exp(-dt / tau)

        prob = attention / max(attention.sum(), 1e-300)

        # 注视位置：注意力加权的高斯混合中采样一个网格中心，再叠加 d_c 尺度抖动
        flat_idx = rng.choice(M * N, p=prob.ravel())
        i, j = np.unravel_index(flat_idx, (M, N))
        cx = x_center[i, j] + rng.normal(0.0, d_c)
        cy = y_center[i, j] + rng.normal(0.0, d_c)

        xs_out.append(cx)
        ys_out.append(cy)

        # 更新注意力：当前位置施加 sigma_s 尺度的高斯核
        dist2 = (x_center - cx) ** 2 + (y_center - cy) ** 2
        attention += np.exp(-dist2 / (2.0 * sigma_s**2))
        last_t = t

    return np.array(xs_out), np.array(ys_out), t_arr


def build_synthetic_a5(seed=0):
    """生成合成 A5 DataFrame，供 load_a5_dataframe 直接消费。"""
    rng = np.random.default_rng(seed)

    rows = []
    emotion_idx = 0
    for emotion in EMOTIONS:
        tau, sigma_s, d_c = (
            TRUE_THETA[emotion_idx * 3],
            TRUE_THETA[emotion_idx * 3 + 1],
            TRUE_THETA[emotion_idx * 3 + 2],
        )
        for trial_i in range(TRIALS_PER_EMOTION):
            trial_number = emotion_idx * TRIALS_PER_EMOTION + trial_i + 1
            x, y, t = _sample_attention_sequence(
                tau, sigma_s, d_c,
                duration_s=PICTURE_DURATION_S,
                sampling_rate=SAMPLING_RATE_HZ,
                seed=seed + trial_number,
            )
            # 映射回屏幕绝对坐标
            gaze_x = x + X_MIN
            gaze_y = y + Y_MIN
            # t_from_picture_start 单位：微秒（与 load_a5_dataframe 一致）
            t_from_picture_start = t * TIMESTAMP_UNIT_SCALE

            for k in range(len(t)):
                row = {
                    "gaze_x": gaze_x[k],
                    "gaze_y": gaze_y[k],
                    "trial_number": trial_number,
                    "t_from_picture_start": t_from_picture_start[k],
                    "positive": 1 if emotion == "positive" else 0,
                    "negative": 1 if emotion == "negative" else 0,
                    "neutral": 1 if emotion == "neutral" else 0,
                    "complex": 1 if emotion == "complex" else 0,
                }
                rows.append(row)
        emotion_idx += 1

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("合成数据演示：情绪注视轨迹参数化测量方法")
    print("=" * 60)

    # 1. 生成合成数据
    a5_df = build_synthetic_a5(seed=0)
    print(f"\n[1] 生成合成 A5 数据：{len(a5_df)} 行，"
          f"{a5_df['trial_number'].nunique()} 个试次")

    # 2. 载入模型数据结构
    data_dict = load_a5_dataframe(
        a5_df,
        timestamp_unit_scale=TIMESTAMP_UNIT_SCALE,
        x_min=X_MIN,
        y_min=Y_MIN,
        width=WIDTH,
        height=HEIGHT,
        resample_ratio=1.0,
    )
    print(f"[2] 载入模型数据结构：{len(data_dict['trials'])} 个试次")

    # 3. 拟合 12 个自由参数
    print("\n[3] 拟合 12 个自由参数 (L-BFGS-B, 有界) ...")
    result = fit_free_parameters(data_dict)

    # 4. 输出恢复参数 vs 真值
    print("\n[4] 参数恢复结果（真值 -> 拟合值）：")
    print(f"    {'情绪':<10} {'参数':<8} {'真值':>10} {'拟合值':>10}")
    print("    " + "-" * 46)
    for ei, emotion in enumerate(EMOTIONS):
        for pi, pname in enumerate(["tau", "sigma_s", "d_c"]):
            true_val = TRUE_THETA[ei * 3 + pi]
            fit_val = result["theta_opt"][ei * 3 + pi]
            print(f"    {emotion:<10} {pname:<8} {true_val:>10.4f} {fit_val:>10.4f}")

    # 5. 报告量
    print("\n[5] 报告量（T_half / H_bar_norm / I_focus）：")
    metrics = result["interpretable_metrics"]
    print(f"    {'情绪':<10} {'T_half(s)':>12} {'H_bar_norm':>12} {'I_focus':>12}")
    print("    " + "-" * 50)
    for emotion in EMOTIONS:
        m = metrics[emotion]
        print(f"    {emotion:<10} {m['T_half']:>12.4f} "
              f"{m['H_bar_norm']:>12.4f} {m['I_focus']:>12.4f}")

    print("\n" + "=" * 60)
    print("完成。上述输出可作为预期输出，供读者核对。")
    print("=" * 60)


if __name__ == "__main__":
    main()
