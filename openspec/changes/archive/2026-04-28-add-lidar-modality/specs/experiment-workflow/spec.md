## ADDED Requirements

### Requirement: LiDAR 配置驱动实验
项目 MUST 支持通过配置文件启动 LiDAR-only 训练和评估。LiDAR-only 配置 MUST 使用 `experiment.task: lidar`，并通过统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和指标流程运行。

#### Scenario: 使用配置启动 LiDAR-only 训练
- **WHEN** 用户通过新 CLI 传入 LiDAR-only 训练配置
- **THEN** 系统 MUST 构建包含 LiDAR 输入的 dataset、配置指定的 LiDAR teacher/student 模型、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求图像、雷达或 GPS 输入
- **AND** LiDAR 输入 MUST 使用 BEV 张量格式

#### Scenario: 使用配置启动 LiDAR-only 评估
- **WHEN** 用户通过新 CLI 传入 LiDAR-only 评估配置和 LiDAR 模型权重
- **THEN** 系统 MUST 构建配置指定的 LiDAR 模型并只使用 LiDAR 输入完成评估
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标

### Requirement: LiDAR fusion 配置驱动实验
项目 MUST 支持通过 fusion `modalities` 配置启用 LiDAR。包含 LiDAR 的 fusion 配置 MUST 复用统一 fusion 训练和评估流程。

#### Scenario: 使用配置启动 image+radar+gps+lidar fusion 训练
- **WHEN** 用户通过训练入口传入 `modalities: ["image", "radar", "gps", "lidar"]` 的 fusion 配置
- **THEN** 系统 MUST 构建四个模态输入所需的 dataset 字段和 fusion teacher/student 模型
- **AND** 系统 MUST 在 batch 准备阶段构造 image、radar、gps 和 lidar 输入

#### Scenario: 使用配置启动 LiDAR 参与的双模态 fusion 训练
- **WHEN** 用户通过训练入口传入包含 `lidar` 的任意合法双模态 fusion 配置
- **THEN** 系统 MUST 只准备 `modalities` 中列出的模态输入
- **AND** 未启用的模态字段 MUST 不影响训练启动

### Requirement: LiDAR 默认实验配置
项目 MUST 提供 LiDAR-only no-KD、LiDAR student no-KD、LiDAR logits KD、LiDAR RKD 和包含 LiDAR 的 fusion 示例配置。所有默认 LiDAR teacher/student 配置 MUST 使用 `gru_params: [64, 64, 2]`。

#### Scenario: LiDAR 默认配置可构建
- **WHEN** 开发者加载 `configs/lidar/*.yaml`
- **THEN** 系统 MUST 能构建对应 dataset、model、loss、distiller、optimizer 和 scheduler
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 2]`

#### Scenario: LiDAR fusion 示例配置可构建
- **WHEN** 开发者加载包含 LiDAR 的 `configs/fusion/*.yaml`
- **THEN** 系统 MUST 能构建对应 fusion teacher 和 fusion student
- **AND** fusion `modalities` MUST 只包含合法模态名称

### Requirement: LiDAR 预处理入口
预处理 CLI MUST 支持生成带 LiDAR 路径列的序列 CSV，并支持按配置离线生成 LiDAR BEV 缓存。

#### Scenario: 生成 LiDAR 序列 CSV
- **WHEN** 用户运行预处理入口并选择 sequence CSV 生成且启用 LiDAR
- **THEN** 系统 MUST 输出包含 `lidar1..lidarN` 的 train/test 序列 CSV

#### Scenario: 生成 LiDAR BEV 缓存
- **WHEN** 用户运行预处理入口并选择 LiDAR BEV 缓存生成
- **THEN** 系统 MUST 根据配置读取点云、应用裁剪和 BEV 构造，并写出可被 dataset 读取的 `.npy` 缓存

### Requirement: LiDAR dry-run 训练
项目 MUST 提供可在小数据或 fixture 上运行的 LiDAR smoke test 路径，用于验证 LiDAR forward、loss、backward、validation 和 checkpoint 保存。

#### Scenario: LiDAR dry-run 训练
- **WHEN** 开发者使用 synthetic、fixture 或小比例数据运行一次 LiDAR 短训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存的核心路径
