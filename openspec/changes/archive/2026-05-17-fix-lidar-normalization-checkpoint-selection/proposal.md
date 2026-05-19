## Why

当前 LiDAR 配置同时出现 `lidar_normalize: false` 和 `lidar_normalization.enabled: true` 时，dataset 实际优先采用 `lidar_normalization` 字典并启用 streaming z-score，导致用户以为在使用 raw BEV，训练却在使用归一化后的稀疏高幅值 BEV。该行为足以解释 LiDAR-only 或含 LiDAR fusion 出现训练集上升、验证集退化的负迁移现象，也会污染后续 teacher registry。

此外，teacher registry 和训练归档默认偏向 `best_top1.pth`，对 LiDAR 这类 Top-1 可能等同多数类水平的 run，会把不符合早停目标的 checkpoint 作为 teacher 候选。

## What Changes

- 修正 LiDAR normalization 配置解析：`lidar_normalization` 只有在用户显式配置或默认值明确要求时才覆盖 `lidar_normalize`，并保持“未显式启用则 raw BEV”的既有 spec 语义。
- 修正 LiDAR modality 默认配置，避免 `lidar_normalize: false` 与 `lidar_normalization.enabled: true` 同时出现在默认合并结果中。
- 保留需要 streaming stats 的 LiDAR baseline/profile 显式启用能力，避免修复默认值时破坏已有可复现实验。
- 调整 teacher registry/checkpoint 选择策略，使 LiDAR teacher 不再默认优先 `best_top1.pth`；未显式指定 checkpoint 时应优先使用早停目标对应的 `best.pth` 或 registry metadata 中声明的 objective checkpoint。
- 扩展 LiDAR 输入质量诊断，记录 raw BEV 稀疏度和归一化后统计，避免 z-score 后 `zero_ratio=0` 掩盖原始 BEV 约 95% 稀疏的问题。
- 增加覆盖测试，锁定配置优先级、checkpoint 解析顺序和 LiDAR quality summary 的 raw/normalized 双视角行为。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `lidar-preprocessing`: 明确 LiDAR normalization 配置优先级、默认 raw BEV 行为，以及质量诊断必须保留 raw BEV 稀疏度。
- `experiment-artifact-registry`: 明确 teacher registry/checkpoint 默认选择不得用多数类 Top-1 checkpoint 覆盖早停目标 checkpoint，尤其是 LiDAR teacher。

## Impact

- 受影响代码：`src/kd_sensing/modalities.py`、`src/kd_sensing/data/datasets/deepsense6g.py`、`src/kd_sensing/engine/data_factory.py`、`src/kd_sensing/evaluation/lidar_diagnostics.py`、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/utils/teacher_registry.py`。
- 受影响配置：启用 LiDAR 但未显式开启 `data.dataset.lidar_normalization.enabled` 的训练将回到 raw BEV；需要 streaming stats 的 baseline/profile 必须显式设置该字段。
- 受影响产物：`final_config.yaml`、`metrics.json`、`train_log.json`、checkpoint registry sidecar 和 `teacher_registry.json` 将更准确记录 LiDAR normalization/profile 和 checkpoint 来源。
- 依赖不变；主要风险是旧 run 的 registry 可能需要重建，旧 LiDAR cache 在 normalization 或 ROI/FoV 改动后应按现有 cache 参数隔离策略重建或避用。
