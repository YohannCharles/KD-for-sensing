# original-code-compatibility Specification

## Purpose
TBD - created by archiving change align-original-code-compatibility. Update Purpose after archive.
## Requirements
### Requirement: 原代码兼容与单模态一致性配置矩阵
项目 MUST 不再为依赖 image motion mask、单通道 motion image branch 或旧 image motion checkpoint 的上游 image-only 与 image+radar 实验提供运行兼容。仍保留的默认 image 和包含 image 的 fusion 配置 MUST 使用 RGB/ImageNet image 输入和兼容的三通道 encoder。项目新增的 radar-only、GPS-only 和 LiDAR-only 单模态配置没有上游原代码基准时，MUST 在共享字段上与当前 RGB image 单模态配置保持一致。

#### Scenario: legacy image motion 配置被拒绝
- **WHEN** 开发者加载依赖 `motion_mask`、`motion_cnn`、`legacy_motion_cnn` 或 `image_motion_*` 字段的旧 image-only 配置
- **THEN** 配置解析 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明 image motion 兼容路径已删除

#### Scenario: image 单模态复现实验迁移到 RGB
- **WHEN** 开发者加载 `configs/image/*.yaml` 中的默认 image 训练配置
- **THEN** 配置 MUST 使用 RGB/ImageNet image 输入
- **AND** 构建出的 image teacher/student 模型 MUST 使用兼容三通道 image 输入的 encoder
- **AND** 配置 MUST 不引用旧 image motion checkpoint、cache 或预处理字段

#### Scenario: 新增单模态继承当前 image 参数
- **WHEN** 开发者加载 `configs/radar/*.yaml`、`configs/gps/*.yaml` 或 `configs/lidar/*.yaml` 中的 teacher no-KD、student no-KD、logits KD 或 RKD 配置
- **THEN** 配置中的共享训练字段 MUST 与当前 `configs/image/` 下同角色配置一致
- **AND** 共享训练字段 MUST 至少包含 `experiment.seed`、`data.dataloader.train_batch_size`、`data.dataloader.test_batch_size`、`data.dataloader.num_workers`、`training.epochs`、`training.lr`、`training.weight_decay`、`training.grad_clip`、`training.patience`、`training.use_early_stopping`、`training.min_delta`、`scheduler.T_0`、`scheduler.T_mult` 和 `scheduler.eta_min`
- **AND** 模态必要差异允许保留，包括 dataset 字段、模型注册名、输入通道、GPS/LiDAR 专用参数、`output.run_name` 和 teacher checkpoint 名称

#### Scenario: image+radar fusion 使用 RGB image branch
- **WHEN** 开发者加载默认 image+radar fusion 配置
- **THEN** fusion teacher/student 的 image branch MUST 接收 RGB/ImageNet image tensor
- **AND** fusion 配置 MUST 不隐式选择单通道 motion mask image branch
- **AND** 默认 checkpoint 来源 MUST 不指向旧 image motion checkpoint

#### Scenario: teacher 权重默认路径可解析
- **WHEN** 用户运行目标 KD 配置且未覆盖 teacher 权重来源
- **THEN** image-only KD 配置 MUST 解析到当前 RGB image teacher 权重位置或要求用户重新训练后提供该权重
- **AND** radar/GPS/LiDAR 单模态 KD 配置 MUST 解析到同模态 teacher no-KD 输出中的 teacher 权重
- **AND** image+radar KD 配置 MUST 解析到当前 RGB image+radar fusion teacher 权重位置或要求用户重新训练后提供该权重

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
目标路径 MUST 显式约束 RGB/ImageNet image 输入尺寸为 `224x224`，radar RA/DA 输入尺寸为 `128x64`。如果配置暴露了其它尺寸但当前模型结构不能安全支持，系统 MUST 拒绝构建或在配置校验阶段给出明确错误。

#### Scenario: image size 不兼容
- **WHEN** 用户在 image-only 或包含 image 的 fusion 配置中设置 `data.dataset.image_size` 不是 `[224, 224]`
- **THEN** 系统 MUST 拒绝运行
- **AND** 错误信息 MUST 说明当前 RGB/ImageNet image encoder 要求 `224x224`

#### Scenario: radar size 不兼容
- **WHEN** 用户在目标兼容 radar-only 或包含 radar 的 fusion 配置中提供非 `128x64` 的 radar RA/DA 输入尺寸
- **THEN** 系统 MUST 拒绝运行
- **AND** 错误信息 MUST 说明当前 radar branch 要求 RA/DA 尺寸为 `128x64`

#### Scenario: 文档说明固定尺寸
- **WHEN** 用户阅读 README 或扩展指南中的目标配置说明
- **THEN** 文档 MUST 明确 image `224x224` 和 radar `128x64` 是当前模型的结构性约束
- **AND** 文档 MUST 不暗示 image motion mask 路径仍受支持

### Requirement: ResNet-18 RGB 路径成为当前 image 默认
新增 ResNet-18 RGB image 路径 MUST 作为默认 RGB 实验入口存在。当前 image-only 与包含 image 的 fusion 配置 MUST 使用 RGB/ImageNet preprocessing、3 通道输入和当前 checkpoint registry 语义。

#### Scenario: image 配置使用 RGB/ImageNet
- **WHEN** 开发者加载 `configs/image/teacher_no_kd.yaml`、`configs/image/student_no_kd.yaml`、`configs/image/logits_kd.yaml` 或 `configs/image/rkd.yaml`
- **THEN** 配置解析后的 image profile MUST 为 `rgb_imagenet`
- **AND** 模型 MUST 使用可接收 3 通道 image tensor 的 branch 或 encoder

#### Scenario: image fusion 配置使用 RGB/ImageNet
- **WHEN** 开发者加载包含 image 的 fusion 配置或 `image_radar_*` canonical 配置
- **THEN** 配置解析后的 image profile MUST 为 `rgb_imagenet`
- **AND** fusion teacher/student 的 image branch MUST 接收 3 通道 RGB/ImageNet tensor

### Requirement: ResNet-18 配置不复用旧 checkpoint
系统 MUST 避免把旧 image checkpoint 静默加载到 ResNet-18 RGB image 模型中，或把 ResNet-18 checkpoint 静默加载到不兼容模型中。结构不匹配时 MUST 沿用严格 checkpoint 加载错误。

#### Scenario: 旧 checkpoint 加载到 ResNet-18 被拒绝
- **WHEN** 用户使用 ResNet-18 RGB image 配置并提供不兼容 checkpoint
- **THEN** 系统 MUST 拒绝静默加载
- **AND** 错误信息 MUST 包含 checkpoint 路径、模型角色和不匹配 key

#### Scenario: ResNet-18 checkpoint 加载到不兼容模型被拒绝
- **WHEN** 用户使用不兼容 image 配置并提供 ResNet-18 RGB image checkpoint
- **THEN** 系统 MUST 拒绝静默加载
- **AND** 错误信息 MUST 包含 checkpoint 路径、模型角色和不匹配 key

