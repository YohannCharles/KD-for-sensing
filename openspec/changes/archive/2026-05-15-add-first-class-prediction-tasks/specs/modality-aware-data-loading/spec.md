## ADDED Requirements

### Requirement: Objective-aware 数据目标加载
数据加载流程 MUST 根据 `experiment.objective` 和模型需求启用对应 targets。未被当前 objective 或显式辅助配置使用的 targets MUST 不被强制读取或拟合 artifact。

#### Scenario: beam objective 不读取辅助目标
- **WHEN** `experiment.objective` 为 `beam` 且未显式启用辅助监督
- **THEN** dataset MUST 不要求 `occlusion_target` 或 `position_target`
- **AND** dataset MUST 不拟合遮挡阈值或位置 target scaler

#### Scenario: occlusion objective 启用遮挡目标
- **WHEN** `experiment.objective` 为 `occlusion`
- **THEN** dataset MUST 返回 `occlusion_label` 和 `occlusion_valid`
- **AND** dataset MUST 拟合或复用训练 split 的遮挡阈值 artifact

#### Scenario: position objective 启用位置目标
- **WHEN** `experiment.objective` 为 `position`
- **THEN** dataset MUST 返回 `position_target` 和 `position_valid`
- **AND** dataset MUST 拟合或复用训练 split 的位置 target scaler artifact，除非配置禁用 target normalization

#### Scenario: multitask objective 启用全部目标
- **WHEN** `experiment.objective` 为 `multitask`
- **THEN** dataset MUST 返回 beam、occlusion 和 position 目标所需的所有字段
- **AND** dataset MUST 保存和复用遮挡阈值与位置 target scaler artifacts

### Requirement: Objective-aware batch 准备
batch/runtime helper MUST 能把当前 objective 所需 targets 搬到目标 device，并保持无效位置 mask 与预测 horizon 对齐。

#### Scenario: occlusion batch targets
- **WHEN** batch 包含遮挡标签且 objective 需要遮挡目标
- **THEN** runtime MUST 返回 device 上的 `occlusion_label` 和 `occlusion_valid`
- **AND** 返回张量 MUST 裁剪或校验到 `num_pred` horizon

#### Scenario: position batch targets
- **WHEN** batch 包含位置目标且 objective 需要位置目标
- **THEN** runtime MUST 返回 device 上的 `position_target` 和 `position_valid`
- **AND** 返回张量 MUST 裁剪或校验到 `num_pred` horizon

#### Scenario: 缺失目标字段
- **WHEN** 当前 objective 需要某个 target 但 batch 缺少对应字段
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指明缺失字段和当前 `experiment.objective`
