## ADDED Requirements

### Requirement: CMSBL 默认关闭并严格恢复 BCACL U2

系统 MUST 提供顶层 `cmsbl.enabled` 开关且默认 false。关闭时系统 MUST 保持 BCACL U2 的模型参数、state dict、forward、private/shared/fusion/restoration loss、mask 采样、checkpoint 和评估行为不变。

#### Scenario: 关闭 CMSBL 运行 U2

- **WHEN** BCACL private/shared head 开启且 `cmsbl.enabled=false`
- **THEN** 同 checkpoint 和输入的 forward 与 total loss MUST 在既有数值容差内等于变更前 U2
- **AND** 模型 MUST 不增加 CMSBL 参数或 buffer

### Requirement: 辅助监督只使用线性 epoch 调度

系统 MUST 为 private/shared 监督分别提供 start weight、end weight、start epoch 和 end epoch，并 MUST 使用一基 epoch执行区间 clamp 和线性插值。`start_weight=end_weight` MUST 表达常量权重，系统 MUST 不提供额外 schedule 类型。

#### Scenario: 调度跨边界

- **WHEN** epoch 位于 decay 开始前、中间和结束后
- **THEN** 权重 MUST 分别等于 start、线性插值和 end
- **AND** checkpoint 恢复到 epoch N+1 时 MUST 使用 N+1 的权重

### Requirement: 容量缺口只使用固定 standalone Top-1 参考

系统 MUST 在训练开始时读取一次包含 dataset、source split、metric、四模态 Top-1 和来源 SHA 的 JSON。系统 MUST 拒绝 outer/test 来源、身份不匹配、缺失模态或非有限值，并 MUST 不支持运行期 validation/test 反馈或替代 reference mode。

#### Scenario: 加载固定容量参考

- **WHEN** stats JSON 声明匹配的 inner train/development Top-1 与四模态身份
- **THEN** reference MUST 在训练期间冻结
- **AND** checkpoint MUST 保存并恢复相同 identity

### Requirement: 容量权重只补足未达到自身参考的模态

系统 MUST 由 train-only 逐模态 Top-1 epoch EMA 计算 `max(0,C-A)/(C+eps)`，并在配置上下界内形成 private/shared 权重。warmup、无有效统计、达到或超过 reference 的模态 MUST 使用中性权重 1。

#### Scenario: 模态低于容量

- **WHEN** 一个模态的有效 train EMA 低于其固定 reference
- **THEN** 其权重 MUST 大于 1 且不超过 max weight
- **AND** 其他 observed 模态的归约 MUST 不因缺失模态数量改变尺度

### Requirement: 困难 mask 只重加权 fusion 与 restoration loss

系统 MUST 将 `image/radar/gps/lidar` 映射到 bit 0--3，并仅使用实际 fusion mask 的加权前 per-sample fusion CE 与 BPA restoration 更新 15-mask EMA/count。系统 MUST 保持既有 600-entry sampling panel、随机流、Router、superset 和 private/shared loss 不变。

#### Scenario: 计算训练 mask 权重

- **WHEN** warmup 后一个充分计数 mask 的 raw loss EMA 高于其他 mask
- **THEN** 其 clipped/mean-normalized 权重 MUST 更高
- **AND** warmup 或低计数 mask 权重 MUST 为 1

### Requirement: CMSBL 长期状态只由训练更新并可恢复

系统 MUST 在 extension checkpoint state 中保存 capacity identity/EMA/initialized 与 mask loss EMA/count/initialized。validation、test 和 fixed-mask evaluator MUST 不更新这些状态；epoch accumulator MUST 在恢复后从空状态开始。

#### Scenario: resume 后继续训练

- **WHEN** 训练从 epoch-end checkpoint 恢复
- **THEN** 长期状态 MUST 与保存前逐值一致
- **AND** 下一 epoch MUST 继续使用恢复后的 capacity 与 mask weights

### Requirement: CMSBL 只提供 V0--V4 最小消融

系统 MUST 将 V0 定义为 U2，V1 为线性调度，V2 为容量权重，V3 为困难 mask loss，V4 为三者组合。系统 MUST 不提供 residual distillation、V5/V6、sampling reweighting 或额外 schedule/capacity mode。

#### Scenario: 比较最小矩阵

- **WHEN** 维护者通过现有训练入口构造 CMSBL 消融
- **THEN** 每个 variant MUST 只改变其声明的训练 objective
- **AND** 不得自动启动 outer test、多 seed 或正式 claim 更新

### Requirement: CMSBL 诊断保持单一事实来源

系统 MUST 每 epoch 写一个结构化 JSON，并将关键标量交给现有 TensorBoard 日志。JSON MUST 包含实际 auxiliary weights、capacity/current/EMA/gap/weight、15-mask count/raw EMA/weight 和 train-only/claim-ineligible 身份；系统 MUST 不维护可由 JSON 派生的重复 CSV。

#### Scenario: 完成训练 epoch

- **WHEN** CMSBL epoch 状态更新完成
- **THEN** JSON 与 checkpoint MUST 反映同一长期状态
- **AND** validation/test MUST 不产生训练态诊断更新
