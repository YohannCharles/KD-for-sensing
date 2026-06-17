## ADDED Requirements

### Requirement: Predictive Robustness 主场景
系统 MUST 提供 Predictive Robustness benchmark suite，用于评估当前图像不可观测或 GPS 可信但错误时，JEPA 是否能利用历史视觉上下文和预测 latent 保持 beam prediction 性能。该 suite MUST 不替代现有 Scenario D CxD；现有 C0-C4 x D0-D7 matrix MUST 继续作为 overall sanity。

#### Scenario: 解析 canonical predictive robustness suite
- **WHEN** benchmark manifest 引用 `predictive_jepa_robustness` canonical preset
- **THEN** 系统 MUST 至少支持 `P0_clean_current`、`P1_current_frame_missing_history_available`、`P2_semantic_occlusion_history_available`、`P3_plausible_wrong_gps_current_image`、`P4_joint_predictive_recovery` 和 `P5_novel_weather_history_available`
- **AND** 每个 condition MUST 记录 image/GPS operator 参数、history window、seed、split、difficulty digest 和 replay metadata
- **AND** 所有 condition MUST 保持 `target_beam`、`beam_power`、sample id 和 split metadata 不变

#### Scenario: Predictive suite 不替换 CxD sanity
- **WHEN** predictive robustness benchmark 完成
- **THEN** 输出 MUST 同时记录 predictive robustness regional metrics 和可选 overall CxD sanity metrics
- **AND** report MUST 明确 predictive metrics 是主 claim 口径，overall CxD 是泛化 sanity

### Requirement: JEPA predictive hybrid fusion 模型组
系统 MUST 支持一个 JEPA predictive hybrid fusion 模型组，用于比较 Image CNN+GPS、现有 JEPA baselines 与新增预测式 JEPA 架构。该模型组 MUST 基于模块化组件实现，并 MUST 不要求恢复旧 KD、HiST 或 residual 研究线。

#### Scenario: 模型组可由配置声明
- **WHEN** 配置声明 `model_group: jepa_predictive_hybrid` 或等价 metadata
- **THEN** 系统 MUST 能构建 JEPA context image encoder、hybrid residual query pooler、temporal predicted latent branch、feature-consistency gate 和 beam head
- **AND** 模型输出 MUST 继续包含现有 beam logits、Top-K/DBA 可评价字段和 runtime metadata
- **AND** 默认 Image CNN+GPS、JEPA GPS-biased 和 JEPA GPS-query-pool 配置 MUST 不被静默替换

#### Scenario: Gate 不读取 condition id
- **WHEN** predictive hybrid 模型在 Predictive Robustness 或 CxD benchmark 中 forward
- **THEN** feature-consistency gate MUST NOT 直接消费 `c_idx`、`d_idx`、`predictive_condition_id` 或 condition string
- **AND** gate diagnostics MUST 说明权重来自 latent consistency、valid masks、observability score、GPS delay/reliability 或等价特征信号

### Requirement: 5 个百分点 claim 口径
系统 MUST 为 predictive robustness 主 claim 提供严格可比较的 margin-vs-CNN 口径。只有在 split、sample_count、label space、metric profile、difficulty digest 和 seed 可比时，系统 MAY 将 `jepa_predictive_hybrid` 相对 Image CNN+GPS 的 predictive DBA margin 标记为真实 claim。

#### Scenario: 计算 predictive DBA margin
- **WHEN** benchmark 同时包含 `jepa_predictive_hybrid` 与 Image CNN+GPS strict comparable rows
- **THEN** 系统 MUST 计算 `predictive_dba`、`predictive_top1`、`cnn_predictive_dba`、`margin_vs_cnn_dba` 和 `claim_pass_5pt`
- **AND** `claim_pass_5pt` MUST 仅在 `margin_vs_cnn_dba >= 0.05` 且 comparability status 为 strict 时为 true

#### Scenario: smoke 不升级为真实 claim
- **WHEN** benchmark 使用 synthetic metrics、mock weights、partial model set 或 missing checkpoint
- **THEN** 系统 MUST 将 claim status 标记为 `mock/smoke`、`pending` 或 `unavailable`
- **AND** docs 和 result claims registry MUST 不记录真实性能数值

### Requirement: Predictive Robustness 输出产物边界
Predictive Robustness workflow MUST 将真实训练、评估和分析产物写入 ignored `outputs/`、`logs/` 或 manifest 指定目录。源码变更 MUST 只包含实现、测试、配置、OpenSpec 和文档账本摘要。

#### Scenario: 写出结构化产物
- **WHEN** predictive robustness benchmark 完成
- **THEN** 输出目录 MUST 包含 machine-readable manifest、condition-level metrics、regional summary、margin-vs-CNN summary 和 warnings
- **AND** 可选图表 MUST 在 manifest 中登记，但真实 PNG/SVG/PDF 不得提交到源码
