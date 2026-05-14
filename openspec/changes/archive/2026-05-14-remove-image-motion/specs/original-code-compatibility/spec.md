## MODIFIED Requirements

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
