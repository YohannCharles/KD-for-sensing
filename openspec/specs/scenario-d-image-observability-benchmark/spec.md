# scenario-d-image-observability-benchmark Specification

## Purpose
定义 Scenario D 图像可观测性等级、与 Scenario C GPS quality axis 的 CxD benchmark 矩阵、可比较模型组、结构化 metrics/plots 输出和 ignored runtime artifact 边界。
## Requirements
### Requirement: Scenario D 图像可观测性等级
系统 MUST 提供 Scenario D 图像可观测性等级，用于以固定 condition id 描述 clean image、天气退化、低光、运动模糊、局部遮挡、帧 dropout、burst missing 和联合 worst-case。等级 MUST 不改变 target label、beam power、sample id、split metadata 或 GPS 输入，除非 Cx-Dy benchmark 同时启用独立的 Scenario C GPS condition。

#### Scenario: 解析固定 D-level preset
- **WHEN** benchmark manifest 引用 canonical Scenario D preset
- **THEN** 系统 MUST 支持 `D0_full_image`、`D1_weather`、`D2_low_light`、`D3_motion_blur`、`D4_partial_occlusion`、`D5_frame_dropout`、`D6_burst_missing` 和 `D7_joint_worst_case`
- **AND** `D0_full_image` MUST 不修改 image 输入
- **AND** `D1_weather` MUST 使用 weather severity sweep 或默认 severity
- **AND** `D2_low_light` MUST 支持 `image_lowlight_prob=0.5`
- **AND** `D3_motion_blur` MUST 支持 `image_blur_prob=0.5`
- **AND** `D4_partial_occlusion` MUST 支持 `image_occlusion_prob=0.5` 和 `image_occlusion_ratio` sweep 或默认值
- **AND** `D5_frame_dropout` MUST 支持 `image_dropout_prob` sweep 或默认值
- **AND** `D6_burst_missing` MUST 支持 `image_burst_dropout_prob`、`max_burst_len` sweep 或默认值

#### Scenario: D7 joint worst-case 语义
- **WHEN** manifest 引用 `D7_joint_worst_case`
- **THEN** 系统 MUST 在 image 侧同时启用 partial occlusion 和 burst missing
- **AND** benchmark MUST 将 `C3_random_async` 或 `C4_severe_async` 与 `D7_joint_worst_case` 的组合标记为重点 worst-case
- **AND** D7 的 image 条件 MUST 不自行移动 GPS 或 target

### Requirement: Cx-Dy 二维鲁棒性矩阵
Benchmark MUST 组合 Scenario C GPS quality axis 与 Scenario D image observability axis，生成 `performance[Cx, Dy]` 二维矩阵。矩阵 MUST 对所有模型使用相同 split、label space、metric profile、sample order、C/D condition id 和 corruption seed。

#### Scenario: 生成完整 Cx-Dy grid
- **WHEN** manifest 声明 Scenario D matrix evaluation
- **THEN** 系统 MUST 至少评估 `C0_sync`、`C1_mild_stale`、`C2_low_rate`、`C3_random_async`、`C4_severe_async`
- **AND** 系统 MUST 至少评估 `D0_full_image` 到 `D7_joint_worst_case`
- **AND** 每个模型 MUST 输出 5x8 条 condition-level metric row 或等价矩阵记录

#### Scenario: 模型组严格可比
- **WHEN** manifest 同时包含 GPS-only、CNN+GPS、Image-AE+GPS、Image-JEPA only 和 Image-JEPA+GPS 模型组
- **THEN** benchmark MUST 校验这些模型的 split、label space、metric profile 和 sample_count 可比较
- **AND** 不可比较模型 MUST 被拒绝写入同一 strict matrix，或被隔离并在 report 中记录原因

### Requirement: Scenario D 指标和论文产物
Benchmark MUST 输出 Scenario D 的结构化指标和论文图产物。输出 MUST 包含 Top-1、Top-3、DBA、worst-case performance、RSI、phase transition curves、CNN vs JEPA crossing point 和 modality dominance ratio。

#### Scenario: 写出指定结果文件
- **WHEN** Scenario D benchmark 完成
- **THEN** 输出根目录 MUST 包含 `results/scenario_d_image_observability.csv`
- **AND** 输出根目录 MUST 包含 `results/heatmap_cx_dy.npy`
- **AND** 输出根目录 MUST 包含 `plots/robustness_surface.png`
- **AND** 输出根目录 MUST 包含 `plots/phase_transition_curve.png`
- **AND** 输出根目录 MUST 包含 `plots/modality_dominance.png`

#### Scenario: 计算 RSI 和 worst-case
- **WHEN** benchmark 聚合 Cx-Dy metrics
- **THEN** 系统 MUST 计算每个模型的 robustness surface integral
- **AND** 系统 MUST 单独记录 `C4_severe_async + D7_joint_worst_case` 的 worst-case performance
- **AND** 汇总表 MUST 记录 primary metric、Top-1、Top-3、DBA、sample_count、seed 和 clean delta

### Requirement: Scenario D 复现与产物边界
Scenario D benchmark MUST 将真实运行产物写入 ignored 的 `outputs/`、`logs/` 或 manifest 指定目录。Benchmark MUST 记录命令、环境、manifest digest、git status 摘要、模型配置、checkpoint provenance、split metadata、C/D preset、corruption 参数、随机种子、warnings 和文件清单。

#### Scenario: 输出目录不污染源码
- **WHEN** benchmark 生成 CSV、NPY、PNG、report、cache 或 runtime manifest
- **THEN** 这些文件 MUST 位于 ignored output root 下
- **AND** 源码变更 MUST 不要求提交真实 benchmark metrics、plots、checkpoint、cache 或 logs

#### Scenario: no label shift guard
- **WHEN** Scenario D transform 或 Cx-Dy benchmark 运行
- **THEN** 系统 MUST 保持 `target_beam`、`beam_power`、soft target、sample id 和 split metadata 不变
- **AND** 单元测试 MUST 能用 synthetic batch 验证 image/GPS corruption 不会移动 label
