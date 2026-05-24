## Why

当前完整 train split 每个 epoch 都遍历全部样本，五模态或重 I/O 配置下单轮反馈较慢，不利于快速调参、排查配置和观察损失趋势。需要一个可配置、可复现的训练子采样机制，让用户在不改 CSV split、不影响验证集语义的前提下缩短每个 epoch。

## What Changes

- 为训练流程增加可选的 train epoch 子采样配置，支持按比例或固定样本数限制每个 epoch 使用的 train 样本。
- 默认行为保持完整 train split 训练；只有显式启用子采样时才改变每个 epoch 的训练样本覆盖范围。
- 子采样必须可复现，并支持每个 epoch 轮换抽样，使多 epoch 调试可以覆盖不同 train 样本而不是固定小切片。
- 验证/test DataLoader 继续使用完整验证或测试 split，避免快速训练配置污染评估指标含义。
- 运行产物必须记录完整 train split 样本数、每个 epoch 实际训练样本数、抽样策略、seed 和 epoch 轮换信息，便于复现实验和解释指标。
- 不引入新的数据文件格式，不修改既有 CSV split，不改变 dataset 样本张量、label 或模态解析语义。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `training-throughput-optimization`: 增加训练 epoch 级 train 样本子采样配置、可复现轮换抽样和运行元数据记录要求。

## Impact

- 影响训练数据构建和 epoch 训练循环，主要涉及 `kd_sensing.engine.data_factory`、`kd_sensing.engine.trainer`、运行元数据与训练吞吐元数据。
- 影响配置默认值和配置覆盖路径，新增配置应兼容现有 `data.dataloader.*`、`training.epochs`、`experiment.seed` 和 resume 语义。
- 影响测试覆盖，需要新增子采样配置解析、每 epoch sampler 行为、验证 split 不受影响、运行日志/metadata 记录和 resume 稳定性的单元测试。
- 不新增第三方依赖，不要求预处理重新生成 split CSV，不改变现有默认训练命令。
