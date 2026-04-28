# original-code-compatibility Specification

## Purpose
TBD - created by archiving change align-original-code-compatibility. Update Purpose after archive.
## Requirements
### Requirement: 原代码兼容与单模态一致性配置矩阵
项目 MUST 为上游原代码实际覆盖的 image-only 与 image+radar 实验提供原代码兼容配置。项目新增的 radar-only、GPS-only 和 LiDAR-only 单模态配置没有上游原代码基准时，MUST 在共享字段上与 image 单模态配置保持一致。兼容配置 MUST 使用随附 `All_models/params_Image*.txt`、`All_models/params_Both*.txt` 与上游 `train_image.py`、`train_both.py` 中的模型层数、训练超参数、KD 参数、调度器参数和 early stopping 参数。

#### Scenario: 单模态 GRU 层数对齐
- **WHEN** 开发者加载 `configs/image/*.yaml`、`configs/radar/*.yaml`、`configs/gps/*.yaml` 或 `configs/lidar/*.yaml` 中的单模态训练配置
- **THEN** 对应 teacher 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 对应 student 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 构建出的单模态 teacher/student 模型 `GRU.num_layers` MUST 为 1

#### Scenario: image+radar GRU 层数对齐
- **WHEN** 开发者加载 `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml` 或 `configs/fusion/image_radar_*.yaml`
- **THEN** image+radar fusion teacher 配置中的 `gru_params` MUST 为 `[64, 64, 2]`
- **AND** image+radar fusion student 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** image+radar teacher no-KD 配置中作为训练主模型的 `model.student` 若为 `fusion_teacher`，其 `gru_params` MUST 为 `[64, 64, 2]`
- **AND** KD 配置构建出的 frozen teacher `GRU.num_layers` MUST 为 2
- **AND** 被训练 student `GRU.num_layers` MUST 为 1

#### Scenario: image 单模态复现实验超参数对齐
- **WHEN** 开发者加载 `configs/image/teacher_no_kd.yaml`、`configs/image/student_no_kd.yaml`、`configs/image/logits_kd.yaml` 或 `configs/image/rkd.yaml`
- **THEN** 配置 MUST 使用 `train_batch_size: 32`、`test_batch_size: 32`、`num_workers: 8`、`epochs: 100`、`grad_clip: 10.0`、`patience: 20`、`min_delta: 0.0001`、`T_0: 10`、`T_mult: 2` 和 `eta_min: 1e-6`
- **AND** 配置 MUST 使用 `experiment.seed: 42`，除非对应来源参数文件显式记录了其它 seed
- **AND** 每个 teacher no-KD、student no-KD、logits KD 和 RKD 配置 MUST 使用对应 `params_Image*.txt` 中记录的 `lr`、`weight_decay`、`temperature`、`alpha`、`alpha_warmup_epochs`、`rkd_pairs_per_anchor`、`rkd_distance_weight` 和 `rkd_angle_weight`

#### Scenario: 新增单模态继承 image 参数
- **WHEN** 开发者加载 `configs/radar/*.yaml`、`configs/gps/*.yaml` 或 `configs/lidar/*.yaml` 中的 teacher no-KD、student no-KD、logits KD 或 RKD 配置
- **THEN** 配置中的共享训练字段 MUST 与 `configs/image/` 下同角色配置一致
- **AND** 共享训练字段 MUST 至少包含 `experiment.seed`、`data.dataloader.train_batch_size`、`data.dataloader.test_batch_size`、`data.dataloader.num_workers`、`training.epochs`、`training.lr`、`training.weight_decay`、`training.grad_clip`、`training.patience`、`training.use_early_stopping`、`training.min_delta`、`scheduler.T_0`、`scheduler.T_mult` 和 `scheduler.eta_min`
- **AND** KD 配置中的共享蒸馏字段 MUST 与 `configs/image/logits_kd.yaml` 或 `configs/image/rkd.yaml` 中同角色配置一致
- **AND** 模态必要差异 MAY 保留，包括 dataset 字段、模型注册名、输入通道、GPS/LiDAR 专用参数、`output.run_name` 和 teacher checkpoint 名称

#### Scenario: image+radar fusion 复现实验超参数对齐
- **WHEN** 开发者加载 `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml` 或 `configs/fusion/image_radar_*.yaml`
- **THEN** 配置 MUST 使用 `train_batch_size: 32`、`test_batch_size: 32`、`num_workers: 8`、`epochs: 100`、`grad_clip: 10.0`、`patience: 20`、`min_delta: 0.0001`、`T_0: 10`、`T_mult: 2` 和 `eta_min: 1e-6`
- **AND** 配置 MUST 使用 `experiment.seed: 42`
- **AND** 每个 teacher no-KD、student no-KD、logits KD 和 RKD 配置 MUST 使用对应 `params_Both*.txt` 中记录的 `lr`、`weight_decay`、`temperature`、`alpha`、`alpha_warmup_epochs`、`rkd_pairs_per_anchor`、`rkd_distance_weight` 和 `rkd_angle_weight`

#### Scenario: teacher 权重默认路径可解析
- **WHEN** 用户运行目标兼容 KD 配置且未覆盖 teacher 权重来源
- **THEN** image-only KD 配置 MUST 解析到兼容的一层 GRU image teacher 权重
- **AND** radar/GPS/LiDAR 单模态 KD 配置 MUST 解析到同模态 teacher no-KD 输出中的一层 GRU teacher 权重
- **AND** image+radar KD 配置 MUST 解析到兼容的二层 GRU fusion teacher 权重
- **AND** image-only 与 image+radar 的默认权重路径 MUST 不要求用户手动迁移 `All_models` 中的随附权重文件

### Requirement: Checkpoint 加载可诊断
项目 MUST 默认严格加载 teacher、评估和 resume checkpoint。权重结构不匹配时，系统 MUST 抛出包含 checkpoint 路径、模型角色、missing keys 和 unexpected keys 的明确错误；只有用户显式选择非严格加载时，系统 MAY 继续运行。

#### Scenario: teacher 权重结构不匹配
- **WHEN** KD teacher 加载的一层 GRU checkpoint 被用于二层 GRU teacher 配置
- **THEN** 系统 MUST 拒绝静默加载
- **AND** 错误信息 MUST 包含缺失的 `GRU.weight_ih_l1`、`GRU.weight_hh_l1`、`GRU.bias_ih_l1` 或 `GRU.bias_hh_l1` 中至少一个 key

#### Scenario: 评估权重结构不匹配
- **WHEN** 用户使用评估入口加载与当前 `model.student` 结构不匹配的权重
- **THEN** 系统 MUST 拒绝静默加载
- **AND** 错误信息 MUST 指出评估权重路径和不匹配 key

#### Scenario: 显式非严格加载
- **WHEN** 用户通过配置显式请求非严格加载 checkpoint
- **THEN** 系统 MAY 调用非严格加载
- **AND** 系统 MUST 在日志或返回结果中记录 missing keys 和 unexpected keys

### Requirement: 恢复训练
训练入口 MUST 让 `training.resume` 生效，并在恢复时加载 student 模型、optimizer、scheduler、已完成 epoch 和 best validation loss。恢复训练 MUST 继续使用统一输出目录、checkpoint 保存和 early stopping 语义。

#### Scenario: 从 last checkpoint 恢复
- **WHEN** 用户设置 `training.resume: true` 且 `output.run_name` 指向已有运行目录
- **THEN** 系统 MUST 从该运行目录的 `checkpoints/last.pth` 加载 checkpoint
- **AND** 后续训练 MUST 从 checkpoint 中记录的下一轮 epoch 开始
- **AND** optimizer、scheduler 和 best validation loss MUST 被恢复

#### Scenario: 从显式路径恢复
- **WHEN** 用户设置 `training.resume` 为 checkpoint 文件路径
- **THEN** 系统 MUST 从该路径加载 checkpoint
- **AND** `training.start_epoch` MUST 仅在 checkpoint 缺少 epoch 字段时作为兜底

#### Scenario: 恢复路径不存在
- **WHEN** 用户启用 resume 但目标 checkpoint 不存在
- **THEN** 系统 MUST 在训练开始前抛出明确错误
- **AND** 错误信息 MUST 包含尝试恢复的 checkpoint 路径

### Requirement: 目标兼容固定输入尺寸约束
目标兼容路径 MUST 显式约束 image 输入尺寸为 `224x224`，radar RA/DA 输入尺寸为 `128x64`。如果配置暴露了其它尺寸但当前模型结构不能安全支持，系统 MUST 拒绝构建或在配置校验阶段给出明确错误。

#### Scenario: image size 不兼容
- **WHEN** 用户在目标兼容 image-only 或包含 image 的 fusion 配置中设置 `data.dataset.image_size` 不是 `[224, 224]`
- **THEN** 系统 MUST 拒绝运行
- **AND** 错误信息 MUST 说明当前 image teacher/fusion teacher 与 motion mask 路径要求 `224x224`

#### Scenario: radar size 不兼容
- **WHEN** 用户在目标兼容 radar-only 或包含 radar 的 fusion 配置中提供非 `128x64` 的 radar RA/DA 输入尺寸
- **THEN** 系统 MUST 拒绝运行
- **AND** 错误信息 MUST 说明当前 radar branch 要求 RA/DA 尺寸为 `128x64`

#### Scenario: 文档说明固定尺寸
- **WHEN** 用户阅读 README 或扩展指南中的目标兼容配置说明
- **THEN** 文档 MUST 明确 image `224x224` 和 radar `128x64` 是当前目标兼容模型的结构性约束
- **AND** 文档 MUST 不暗示这些配置已支持任意 image/radar 尺寸

