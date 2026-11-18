# emotion-gaze-params

情绪注视轨迹的参数化测量方法（T5 / T6 眼动任务）——论文复现代码。

本仓库是面向 **Behavior Research Methods (BRM)** 投稿的可复现代码存档，包含：

1. **实验范式运行代码**（`task/`）：单幅情绪图片任务（T5）与多幅情绪图片任务（T6）的刺激呈现程序；
2. **参数拟合核心**（`fit/`）：从原始注视轨迹估计 12 个自由参数（每类情绪 × tau / sigma_s / d_c），并计算报告量；
3. **参数恢复验证**（`recovery/`）：用合成数据验证拟合流程能否找回已知参数；
4. **联合分析脚本**（`analysis/`）：PCA、量表关联、行为锚定等下游分析；
5. **可运行示例**（`example/`）：不含真实被试的合成数据演示 + 预期输出。

> **数据说明**：原始眼动数据包含参与者可识别信息，因隐私与数据保护规则不予公开。本仓库通过 `example/generate_synthetic_demo.py` 提供完全合成的模拟注视轨迹，演示完整拟合与报告量计算流程。

---

## 目录结构

```
emotion-gaze-params/
├── task/                          # 实验范式运行代码（采集阶段）
│   ├── t5_emotional_orienting/    # 单幅情绪图片任务（T5）
│   │   ├── main.py                #   刺激呈现主程序
│   │   ├── TrialList/             #   试次列表（刺激编号 + 时长分配）
│   │   │   ├── trials_list.csv
│   │   │   └── 情绪定向任务_trigger.csv
│   │   └── readme.txt
│   └── t6_emotional_observation/  # 多幅情绪图片任务（T6）
│       ├── main.py
│       ├── TrialList/
│       │   ├── trial_list.csv
│       │   └── 情绪自由搜索_trigger_含义.csv
│       └── readme.txt
├── fit/                           # 拟合核心
│   ├── t5_memory/                 # T5 模型拟合与数据流水线
│   │   ├── model.py               #   12 参数模型、似然、拟合、报告量
│   │   ├── pipeline.py            #   A1→A5 数据流水线
│   │   ├── feature_extraction.py
│   │   ├── clean_input.py
│   │   └── test_binocular_validity.py
│   └── t6/                        # T6 拟合流水线
│       └── pipeline.py
├── recovery/                      # 参数恢复验证
│   ├── run_t5_parameter_recovery.py
│   ├── run_t6_formal_recovery.py
│   ├── design_t6_formal_recovery.py
│   ├── build_field_common_recovery_framework.py
│   └── parameter_recovery_common.py
├── analysis/                      # 下游联合分析（PCA / 量表关联 / 行为锚定等）
└── example/                       # 可运行示例
    ├── generate_synthetic_demo.py #   合成数据生成 + 拟合 + 报告量演示
    └── expected_output.txt        #   预期输出（供读者核对）
```

---

## 快速开始（合成数据演示）

```bash
# 依赖：Python 3.9+，numpy / scipy / pandas
python example/generate_synthetic_demo.py
```

该脚本会：

1. 用一组**已知真值参数**生成完全合成的注视轨迹（不含真实被试数据）；
2. 调用 `fit/t5_memory/model.py` 的拟合入口，从合成数据恢复 12 个自由参数；
3. 打印"真值 vs 拟合值"对照表，以及报告量 `T_half` / `H_bar_norm` / `I_focus`。

输出应与 `example/expected_output.txt` 一致（受随机种子与优化器影响，拟合值在真值附近小幅波动属正常）。

---

## 方法与参数规格（可复现关键项）

### 模型（T5）

- **自由参数**：12 个，对应 4 类情绪（`positive` / `negative` / `neutral` / `complex`）各 3 个：
  - `tau`：注意力衰减时间尺度；
  - `sigma_s`：空间注意扩散尺度；
  - `d_c`：注视位置空间散布尺度。
- **参数顺序**：`[positive(tau, sigma_s, d_c), negative(...), neutral(...), complex(...)]`。

### 拟合上下界（L-BFGS-B 有界优化）

| 参数 | 下界 | 上界 |
|---|---|---|
| `tau`（positive / negative / neutral） | 0.01 | 10.0 |
| `tau`（complex） | 0.0001 | 10.0 |
| `sigma_s`（全部情绪） | 5.0 | 600.0 |
| `d_c`（全部情绪） | 5.0 | 400.0 |

- 初始值 `init_theta = [0.26, 27, 31, 0.18, 38, 32, 0.14, 34, 34, 0.15, 31, 28]`。
- 见 `fit/t5_memory/model.py` 中 `fit_free_parameters()`。

### 抽稀（resample）规则

- 数据流水线支持 `resample_ratio`（`0 < ratio <= 1`），默认 `1.0`（不抽稀）。
- 当 `ratio < 1` 时，每个试次按 `np.linspace(0, n-1, n_keep)` 等距保留 `n_keep = floor(n_total * ratio)` 个采样点（至少保留 1 点）。
- 见 `fit/t5_memory/model.py` 中 `load_a5_dataframe()` 与 `pipeline.py`。

### 时间原点与坐标映射

- 时间单位：`t_from_picture_start` 以**微秒**记录，模型内除以 `timestamp_unit_scale = 1e6` 转为秒；时间原点为**图片阶段首个采样点**。
- 屏幕坐标映射（T5）：`x_min = 480, y_min = 180, width = 960, height = 720`（像素）；图片显示区域 `x ∈ [480, 1440]`, `y ∈ [180, 900]`。
- 采样率：约 200 Hz（`SAMPLING_RATE_HZ`）。

### 报告量

| 报告量 | 定义 | 代码位置 |
|---|---|---|
| `T_half` | 注意力衰减半衰期 `tau · ln 2` | `compute_interpretable_metrics()` |
| `H_bar_norm` | 归一化平均空间熵（除以 `ln(M·N)`，`M=12, N=9`） | 同上 |
| `I_focus` | 单位聚焦点信息量 `1 - exp(-r_focus² / 2·d_c²)`，`r_focus = 50` | 同上 |

---

## 依赖

- Python 3.9+
- numpy ≥ 1.24
- scipy ≥ 1.9
- pandas ≥ 1.5

实验程序（`task/` 下的 `main.py`）依赖 `pygame` 与眼动仪 SDK（Pupil Labs），需在采集硬件环境中运行。

---

## 声明

本仓库代码的 Open Practices / Data availability / Code availability 声明见 `STATEMENTS.md`。
