# 声明（Statements）——供 BRM 投稿使用

本文档给出面向 **Behavior Research Methods (BRM)** 投稿的三份标准声明。对应仓库：

- **代码仓库**：https://github.com/r1ngtone/emotion-gaze-params（tag `v1.0`）

> 说明：三份声明均按 BRM 官方模板组织。带 `【待作者确认】` 的字段需作者核实后填实，其余部分据实撰写，未编造任何批号、联系人或共享结论。

---

## 一、Open Practices Statement

> 位置：正文参考文献部分**之前**。

Materials and analysis code are available in the project repository at https://github.com/r1ngtone/emotion-gaze-params（tag `v1.0`）. Due to data protection rules and participant privacy, the raw eye-tracking data cannot be made publicly available; the analysis code in the repository is documented and illustrated with fully synthetic, simulated data（`example/generate_synthetic_demo.py`，见 `example/expected_output.txt`）. None of the reported studies were preregistered.

- 中文对照（供正文翻译参考，非正式投稿文本）：
  材料与分析代码可在项目仓库 https://github.com/r1ngtone/emotion-gaze-params（tag `v1.0`）获取。因数据保护规则与参与者隐私限制，原始眼动数据不公开；仓库中的分析代码已附文档，并以完全合成的模拟数据演示（`example/generate_synthetic_demo.py`，预期输出见 `example/expected_output.txt`）。本研究未预注册。

---

## 二、Data Availability Statement

> 位置：Declarations 部分。采用 BRM 数据政策模板 3（数据因合理理由不公开）。

The data that support the findings of this study are not publicly available due to participant privacy and data protection rules; they contain personally identifying information. De-identified aggregate parameter tables and a fully synthetic demonstration dataset are available in the project repository（https://github.com/r1ngtone/emotion-gaze-params）. Reasonable requests for additional data may be directed to the corresponding author, subject to the terms of the original informed consent and institutional requirements. `【待作者确认】`：通讯作者姓名与联系方式；"不公开"的最终口径（不可提供 / 可向通讯作者索取）。

- 中文对照：支撑本研究结果的数据因参与者隐私与数据保护规则不公开，其中含可识别个人身份的信息。去标识化的聚合参数表与完全合成的演示数据集见项目仓库（https://github.com/r1ngtone/emotion-gaze-params）。符合原始知情同意与机构要求的前提下，可向通讯作者合理索取更多数据。

---

## 三、Code Availability Statement

> 位置：Declarations 部分。

The custom analysis code used to produce the results reported in this paper is available in the project repository at https://github.com/r1ngtone/emotion-gaze-params（tag `v1.0`），covering the T5/T6 task-presentation programs（`task/`），parameter estimation（`fit/`），parameter-recovery and validation scripts（`recovery/`），and the joint-analysis scripts（`analysis/`）.

- 中文对照：用于产生本文结果的自定义分析代码可在项目仓库 https://github.com/r1ngtone/emotion-gaze-params（tag `v1.0`）获取，涵盖 T5/T6 任务呈现程序（`task/`）、参数估计（`fit/`）、参数恢复与验证脚本（`recovery/`）及联合分析脚本（`analysis/`）。

---

## 附：提交前需作者确认的清单

| # | 待确认项 | 影响哪段 |
|---|---|---|
| 1 | 通讯作者姓名与联系方式 | Data Availability |
| 2 | 数据"不公开"的最终口径（不可提供 / 可向通讯作者索取） | Data Availability |
| 3 | 伦理批准/豁免机构与批号、知情同意方式（稿件第 2.1 节占位） | 全篇（独立于三份声明） |
| 4 | tag `v1.0` 需在首次公开推送后创建并核对 | Code / Open Practices |

已确认项：

- 研究**未预注册**（已定稿）；
- 仓库为**公开仓库**（代码可公开获取，无需审稿访问方案）；
- 数据因隐私**不公开**，代码用**合成数据**演示（`TEST_` 前缀等真实被试数据已确认不在本仓库）。
