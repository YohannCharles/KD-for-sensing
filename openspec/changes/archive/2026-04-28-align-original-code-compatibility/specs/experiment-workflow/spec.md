## MODIFIED Requirements

### Requirement: 训练与评估行为等价
结构重构后，默认 image-only、radar-only、GPS-only、LiDAR-only 和 fusion 工作流 MUST 通过新脚本保持当前算法的核心训练、验证和评估语义，包括默认序列长度、预测步数、类别数、KD 模式、teacher 权重选择、student 架构选择、early stopping、gradient clipping、checkpoint 恢复和指标计算。上游原代码实际覆盖的 image-only 与 image+radar 配置 MUST 按原代码和随附参数文件对齐 GRU 层数与训练超参数；radar-only、GPS-only 和 LiDAR-only 是本项目新增单模态配置，MUST 在共享字段上与 image 单模态配置保持一致。

#### Scenario: 新配置默认参数
- **WHEN** 用户使用新脚本和默认配置启动训练或评估
- **THEN** 系统 MUST 使用从旧实现迁移而来的默认任务语义，并保持相同的任务类型
- **AND** `configs/image/*.yaml`、`configs/radar/*.yaml`、`configs/gps/*.yaml` 和 `configs/lidar/*.yaml` 中的单模态 teacher 与 student `gru_params` MUST 为 `[64, 64, 1]`
- **AND** `configs/radar/*.yaml`、`configs/gps/*.yaml` 和 `configs/lidar/*.yaml` 中的共享训练字段 MUST 与 `configs/image/` 下同角色配置一致
- **AND** `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml` 和 `configs/fusion/image_radar_*.yaml` 中的 image+radar fusion teacher `gru_params` MUST 为 `[64, 64, 2]`
- **AND** `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml` 和 `configs/fusion/image_radar_*.yaml` 中的 image+radar fusion student `gru_params` MUST 为 `[64, 64, 1]`
- **AND** image+radar teacher no-KD 配置中作为训练主模型的 `model.student` 若为 `fusion_teacher`，其 `gru_params` MUST 为 `[64, 64, 2]`
- **AND** `src/kd_sensing/config/defaults.py` MUST 不把所有 teacher/student 的 `gru_params` 统一强制为 `[64, 64, 2]`

#### Scenario: 默认 student 架构与 GRU 层数
- **WHEN** 用户使用默认 image-only、radar-only、GPS-only、LiDAR-only 或 fusion student 实验配置构建模型
- **THEN** 系统 MUST 为 image-only 工作流构建轻量 `image_student`
- **AND** 系统 MUST 为 radar-only 工作流构建轻量 `radar_student`
- **AND** 系统 MUST 为 GPS-only 工作流构建轻量 `gps_student`
- **AND** 系统 MUST 为 LiDAR-only 工作流构建轻量 `lidar_student`
- **AND** 系统 MUST 为 fusion 工作流构建轻量 `fusion_student`
- **AND** image、radar、GPS 和 LiDAR 单模态 student 模型的 `GRU.num_layers` MUST 为 1
- **AND** 原代码兼容 image+radar fusion student 模型的 `GRU.num_layers` MUST 为 1
- **AND** 文档 MUST 说明二层 GRU student 是历史 canonical 配置或特定扩展配置，不是当前单模态和 image+radar 兼容配置的默认结构

#### Scenario: 默认 teacher GRU 层数
- **WHEN** 用户通过目标兼容配置构建 image、radar、GPS、LiDAR 或 image+radar fusion teacher 模型
- **THEN** image、radar、GPS 和 LiDAR 单模态 teacher 模型的 `GRU.num_layers` MUST 为 1
- **AND** image、radar、GPS 和 LiDAR 单模态 teacher 配置 MUST 使用 `gru_params: [64, 64, 1]`
- **AND** image+radar fusion teacher 模型的 `GRU.num_layers` MUST 为 2
- **AND** image+radar fusion teacher 配置 MUST 使用 `gru_params: [64, 64, 2]`

#### Scenario: checkpoint 恢复语义
- **WHEN** 用户在训练配置中启用 `training.resume`
- **THEN** 训练流程 MUST 尝试恢复 checkpoint
- **AND** 恢复 MUST 包含模型权重、optimizer、scheduler、已完成 epoch 和 best validation loss
- **AND** `training.start_epoch` MUST 不再是唯一影响恢复 epoch 的字段

#### Scenario: dry-run 训练
- **WHEN** 开发者使用 synthetic 或小比例数据运行一次短训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存的核心路径
- **AND** 使用目标兼容配置时，smoke test MUST 使用与该配置匹配的 GRU 层数构建模型
