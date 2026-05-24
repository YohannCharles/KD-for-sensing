## ADDED Requirements

### Requirement: 训练 epoch 级 train 子采样
训练流程 MUST 支持显式配置的 train epoch 子采样，使用户能够在保留原 train CSV 和 dataset 语义的前提下限制每个 epoch 实际参与训练的样本数。该能力 MUST 默认关闭；关闭时训练 MUST 继续遍历完整 train split。

#### Scenario: 默认完整 train split
- **WHEN** 用户未启用 `training.epoch_subsampling.enabled`
- **THEN** train DataLoader MUST 按现有行为遍历完整 train dataset
- **AND** 现有训练配置 MUST 不需要修改即可运行

#### Scenario: 按比例限制每个 epoch 样本
- **WHEN** 用户设置 `training.epoch_subsampling.enabled=true` 且提供合法 `fraction`
- **THEN** 每个 train epoch MUST 使用完整 train dataset 中按该比例计算的有效样本数
- **AND** 有效样本数 MUST 至少为 1 且不得超过完整 train dataset 长度

#### Scenario: 按固定数量限制每个 epoch 样本
- **WHEN** 用户设置 `training.epoch_subsampling.enabled=true` 且提供合法 `num_samples`
- **THEN** 每个 train epoch MUST 使用不超过 `num_samples` 的 train 样本
- **AND** 当 `num_samples` 大于完整 train dataset 长度时，系统 MUST 退化为完整 train epoch 并在运行元数据中记录该退化结果

#### Scenario: 子采样配置错误清晰失败
- **WHEN** 用户同时设置 `fraction` 和 `num_samples` 或提供非法比例/数量
- **THEN** 训练启动 MUST 失败并给出包含 `training.epoch_subsampling` 的清晰错误信息
- **AND** 系统 MUST 不静默回退为完整训练

#### Scenario: 每 epoch 可复现轮换抽样
- **WHEN** train epoch 子采样启用且 `rotate_each_epoch=true`
- **THEN** 不同 epoch MUST 基于实验 seed 和 epoch 编号生成可复现的无放回样本选择
- **AND** checkpoint resume 后同一绝对 epoch MUST 生成与未中断运行一致的样本选择

#### Scenario: 固定子集调试
- **WHEN** train epoch 子采样启用且 `rotate_each_epoch=false`
- **THEN** 每个 epoch MUST 使用同一个可复现 train 子集
- **AND** 该子集 MUST 由配置 seed 或 `experiment.seed` 决定

#### Scenario: 验证 split 不受 train 子采样影响
- **WHEN** train epoch 子采样启用并完成一个训练 epoch
- **THEN** 验证或测试 DataLoader MUST 继续使用完整 validation/test split
- **AND** 验证指标 MUST 不因 train 子采样配置而减少评估样本数

#### Scenario: 运行产物记录子采样语义
- **WHEN** train epoch 子采样启用
- **THEN** 最终配置、运行元数据或训练日志 MUST 记录完整 train split 样本数、每个 epoch 有效 train 样本数、抽样方式、seed、`rotate_each_epoch` 和是否退化为完整 epoch
- **AND** epoch 级日志 MUST 能区分完整训练 epoch 与子采样训练 epoch
