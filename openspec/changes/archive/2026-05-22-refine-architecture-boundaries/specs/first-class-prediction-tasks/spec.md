## ADDED Requirements

### Requirement: Prediction objective 元数据与 loss 实现分离
系统 MUST 将 prediction objective 的元数据契约作为轻量能力暴露。元数据 MUST 覆盖 objective 名称、默认主指标、指标方向、可用指标、early stopping alias、required targets、required outputs、history fields、TensorBoard scalar 映射和 runtime metadata。该元数据 MUST 可被配置加载、配置 normalization、训练、验证和评估复用，且读取元数据时 MUST 不导入 torch loss 实现。

#### Scenario: 配置 normalization 读取 objective 元数据
- **WHEN** 配置加载流程需要根据 `experiment.objective` 补齐默认 early stopping metric 和 mode
- **THEN** 系统 MUST 使用轻量 objective 元数据
- **AND** 配置加载流程 MUST 不导入 torch、prediction loss helper 或训练 runtime

#### Scenario: 训练 runtime 复用同一 objective 元数据
- **WHEN** 训练 runtime 写出 final config、history fields、TensorBoard scalar 或 primary metric metadata
- **THEN** 系统 MUST 复用同一 objective 元数据契约
- **AND** 训练 runtime MUST 不维护与配置路径不同的 objective metric 表

#### Scenario: loss helper 保持 runtime 职责
- **WHEN** 系统计算 `beam`、`occlusion`、`position`、`multitask`、`current_beam_selection`、`current_los_classification`、`current_link_quality` 或 `selection_multitask` loss
- **THEN** torch 相关 target 和 loss 计算 MUST 位于 engine runtime 或等价重依赖模块
- **AND** 该模块 MUST 从轻量 objective 元数据读取契约，而不是复制 objective 列表和 metric 表

### Requirement: Objective 元数据公开 API 兼容
系统 MUST 保持现有 objective helper 的公开行为兼容。现有调用方如果从 `kd_sensing.engine.prediction_objectives` 读取 objective metadata helper，仍 MUST 能获得相同语义的返回值；新增轻量模块 MAY 成为内部实现来源，但不得要求用户修改训练配置或 CLI 参数。

#### Scenario: 现有 objective helper 继续可用
- **WHEN** 现有代码调用 `resolve_prediction_objective`、`objective_spec`、`objective_runtime_metadata`、`objective_history_fields` 或 `objective_tensorboard_scalars`
- **THEN** 调用 MUST 继续成功
- **AND** 返回字段和默认值 MUST 与变更前兼容

#### Scenario: 新增 objective 只改一处元数据契约
- **WHEN** 开发者新增 prediction objective 或调整 objective 默认 metric
- **THEN** 主要变更 MUST 位于轻量 objective 元数据契约及必要的 runtime loss/metric 实现
- **AND** 配置 normalization 和训练日志字段 MUST 通过该契约自动消费更新
