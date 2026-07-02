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
- **WHEN** manifest 同时包含 GPS-only、Image ResNet+GPS、Image-AE+GPS、Image-JEPA only 和 Image-JEPA+GPS 模型组
- **THEN** benchmark MUST 校验这些模型的 split、label space、metric profile 和 sample_count 可比较
- **AND** 不可比较模型 MUST 被拒绝写入同一 strict matrix，或被隔离并在 report 中记录原因

### Requirement: Scenario D 指标和论文产物
Benchmark MUST 输出 Scenario D 的结构化指标和论文图产物。输出 MUST 包含 Top-1、Top-3、DBA、worst-case performance、RSI、phase transition curves、ResNet vs JEPA crossing point 和 modality dominance ratio。

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

### Requirement: Scenario D phase transition artifact
Scenario D benchmark MUST 将 CxD matrix 的基础 metrics 扩展为 phase transition artifact。该 artifact MUST 与现有 `results/scenario_d_image_observability.csv` 和 `results/heatmap_cx_dy.npy` 保持兼容，同时新增 phase diagram、crossing、dominance 和 failure decomposition 输出。

#### Scenario: Scenario D benchmark 输出 phase analysis 文件
- **WHEN** Scenario D CxD benchmark 完成至少一个 strict comparable model group
- **THEN** output root MUST 继续写出 `results/scenario_d_image_observability.csv`
- **AND** output root MUST 继续写出 `results/heatmap_cx_dy.npy`
- **AND** output root MUST 写出 `results/cxd_phase_diagram.csv`
- **AND** output root MUST 写出 `results/crossing_region_Cx_Dy.json`
- **AND** output root MUST 写出 `results/failure_mode_decomposition.csv`
- **AND** runner manifest MUST 在 `output_files` 中登记这些文件

#### Scenario: Scenario D smoke 不升级为研究 claim
- **WHEN** Scenario D benchmark 使用 synthetic metrics、mock weights 或 smoke manifest
- **THEN** all phase analysis outputs MUST 标记为 `mock/smoke` 或等价状态
- **AND** claim registry MUST 不记录真实性能数值
- **AND** docs MUST 明确真实 checkpoint matrix 仍需本地运行且输出不进入源码

### Requirement: Scenario D dominance evidence status
Scenario D benchmark MUST 区分 performance metric 与 dominance diagnostic evidence。缺少真实 diagnostics 的模型 MUST 仍可进入 CxD performance heatmap，但 MUST 不被写成有 dominance 解释证据。

#### Scenario: dominance 诊断字段可追踪
- **WHEN** Scenario D CxD rows 被写出
- **THEN** 每个 dominance row MUST 记录 model、group、gps_condition、image_condition、seed、diagnostic_source、diagnostic_status 和 unavailable reason
- **AND** performance-only rows MUST 不自动产生正式 contribution score

#### Scenario: attention 或 gradient 不可用
- **WHEN** 某个模型缺少 attention、gradient 或 latent diagnostics
- **THEN** Scenario D benchmark MUST 将该模型的 dominance status 标记为 unavailable 或 skipped
- **AND** benchmark MUST 继续输出 accuracy heatmap、RSI、worst-case 和 crossing 可用部分
- **AND** report 或 manifest MUST 标记该 dominance evidence 不足以支撑模态主导结论

### Requirement: Scenario D crossing boundary 可比较性
Scenario D benchmark MUST 只在严格可比较的模型和 condition rows 上计算 ResNet vs JEPA crossing boundary。比较 MUST 保持同一 split、label space、metric profile、sample_count、seed 和 difficulty digest。

#### Scenario: 不可比较模型不进入 crossing boundary
- **WHEN** ResNet 与 JEPA 模型的 split、metric profile、sample_count 或 difficulty digest 不一致
- **THEN** benchmark MUST 不把它们写入同一 crossing boundary
- **AND** manifest MUST 记录不可比较字段
- **AND** 对应 crossing result MUST 标记为 unavailable 或 isolated group

#### Scenario: query_pool crossing shift summary
- **WHEN** Scenario D benchmark 同时包含 GPS-biased JEPA 和 GPS-query-pool JEPA
- **THEN** benchmark MUST 输出 query_pool 相对 biased JEPA 的 crossing shift summary
- **AND** summary MUST 使用同一 metric 和同一 CxD grid
- **AND** 缺少任一配对模型时 MUST 标记为 unavailable

### Requirement: Scenario D benchmark suite
JEPA GPS shortcut benchmark MUST 支持 Scenario D image observability suite。Suite MUST 复用 shared difficulty pipeline，且 MUST 能与 existing Scenario C async GPS suite 组合为 Cx-Dy matrix。

#### Scenario: manifest 引用 Scenario D suite
- **WHEN** benchmark manifest 声明 suite type `scenario_d_image_observability`
- **THEN** runner MUST 标准化 D-level condition、image operator 参数、seed 和 output artifact plan
- **AND** runner MUST 将 image corruption 委托给 shared difficulty operator
- **AND** runner MUST 不维护独立平行的 image corruption 实现

#### Scenario: Scenario C 与 D 联合执行
- **WHEN** manifest 声明 joint suite `scenario_c_x_d_image_observability`
- **THEN** runner MUST 对每个模型执行 Scenario C condition 与 Scenario D condition 的笛卡尔组合
- **AND** 每个 row MUST 记录 `gps_condition`、`image_condition`、C severity、D severity、seed 和 difficulty digest

### Requirement: Scenario D required model groups
Benchmark MUST 支持 Scenario D 指定的模型组：GPS-only、Image ResNet+GPS、Image-AE+GPS、Image-JEPA only 和 Image-JEPA+GPS。Runner MUST 将这些模型组映射到现有 config/weights/registry 语义，并 MUST 记录模型是否消费 image/GPS reliability metadata。

#### Scenario: required model group 校验
- **WHEN** manifest 声明 strict Scenario D evaluation
- **THEN** runner MUST 校验 required model groups 是否齐全，或在显式允许 partial run 时记录缺失模型组
- **AND** report MUST 区分 standard fusion、ResNet/AE visual encoder、JEPA visual encoder 和 observability-aware fusion

#### Scenario: Image-JEPA only 不消费 GPS 输入
- **WHEN** model group 为 Image-JEPA only
- **THEN** runner MUST 仍按 Cx-Dy 条件记录 GPS condition metadata 以保持矩阵对齐
- **AND** 模型 forward MUST 不要求 GPS input tensor

### Requirement: Scenario D aggregation 和图表
Benchmark MUST 聚合 Scenario D matrix，并导出 Cx-Dy heatmap、robustness surface、phase transition、ResNet vs JEPA crossing point 和 modality dominance 图表或表格。图表生成失败时，metrics CSV 和 manifest MUST 仍然写出，并记录 warning。

#### Scenario: 输出 Cx-Dy aggregation
- **WHEN** Scenario D matrix 完成至少一个模型
- **THEN** runner MUST 写出包含 model、gps_condition、image_condition、metric、sample_count、seed 和 clean delta 的 long-form CSV
- **AND** runner MUST 写出按模型排序的 heatmap NPY 或等价矩阵 artifact

#### Scenario: attention 不可用时 dominance 降级
- **WHEN** 某个模型不提供 attention 或 fusion weights
- **THEN** modality dominance ratio MUST 使用配置声明的 fallback 或跳过该模型
- **AND** warnings MUST 记录 unavailable reason
