# lidar-modality-model Specification

## Purpose
定义 LiDAR 模态模型、BEV encoder、normalization 和输入质量诊断契约。
## Requirements
### Requirement: LiDARFeatureExtractor 结构
系统 MUST 提供 `LidarFeatureExtractor`，用于从 LiDAR BEV 序列中提取每个时隙的固定长度特征。该 feature extractor MUST 接收形状为 `(batch, sequence, channels, height, width)` 的 LiDAR BEV 张量，并输出 `(batch, sequence, feature_size)`。该类 MAY 通过 `kd_sensing.models.lidar` 或 `kd_sensing.models` 窄导入暴露，但 MUST NOT 作为 current `MODELS` 注册名暴露。

#### Scenario: LiDAR feature extractor 前向输出
- **WHEN** `LidarFeatureExtractor` 接收形状为 `(B, T, C, H, W)` 的 LiDAR BEV 输入
- **THEN** 输出 MUST 为形状 `(B, T, feature_size)` 的特征张量
- **AND** 输出 feature 维 MUST 等于构造参数 `n_feature` 或 `feature_size`

#### Scenario: LiDAR feature extractor 不作为完整模型注册
- **WHEN** 开发者查看 current `MODELS.list()`
- **THEN** 输出 MUST NOT 包含 `lidar_feature_extractor`
- **AND** 需要配置构建 LiDAR encoder 时 MUST 使用 `ENCODERS` 中的 `lidar_cnn`

### Requirement: LiDAR-only 输入准备
系统 MUST 提供 LiDAR-only 输入准备路径，从 batch 中读取 `lidar`，按现有预测窗口规则补齐未来占位帧，并将结果传给 LiDAR 模型。

#### Scenario: 准备 LiDAR-only batch
- **WHEN** 训练、验证或评估流程处理 `experiment.task: lidar`
- **THEN** 系统 MUST 使用 batch 中的 `lidar` 构造 LiDAR 输入
- **AND** 系统 MUST 不要求图像、雷达或 GPS 输入参与模型 forward

#### Scenario: LiDAR 预测窗口对齐
- **WHEN** `seq_length` 为 8 且 `num_pred` 为 3
- **THEN** LiDAR-only 输入 MUST 包含最近 8 个 LiDAR 时隙和 2 个未来 zero padding 时隙
- **AND** 验证和损失计算 MUST 使用最后 `num_pred` 个输出时隙与 `[t+1, t+2, t+3]` 标签对齐
- **AND** 输出时隙对齐 MUST 不包含历史窗口最后一个 beam

### Requirement: LiDAR KD 入口已移除
LiDAR-only 训练 MUST 不再支持 logits KD、RKD 或 distiller 运行时。旧 LiDAR KD 配置路径 MUST 在配置解析阶段失败，并引导用户使用 `configs/lidar/strong.yaml`、`configs/lidar/lightweight.yaml` 或 `configs/lidar/supervised.yaml`。

#### Scenario: LiDAR logits KD 被拒绝
- **WHEN** 用户运行旧 LiDAR-only logits KD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 frozen LiDAR teacher 或 distiller

#### Scenario: LiDAR RKD 被拒绝
- **WHEN** 用户运行旧 LiDAR-only RKD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不计算关系蒸馏损失

### Requirement: LiDAR 单模态默认 GRU 层数
默认 LiDAR teacher 和 LiDAR student 单模态配置 MUST 使用一层 GRU，以便与当前 LiDAR 配置、README 和测试保持一致。

#### Scenario: lidar_teacher 默认 GRU 层数
- **WHEN** 用户通过默认 LiDAR teacher no-KD 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1

#### Scenario: lidar_student 默认 GRU 层数
- **WHEN** 用户通过默认 LiDAR student no-KD 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1

### Requirement: LiDAR baseline 必须报告退化基线对比
LiDAR-only baseline 训练和评估 MUST 报告模型指标与退化基线的对比。退化基线至少 MUST 包含 majority-class baseline；当序列 beam 历史可用时，还 MUST 包含 last-beam baseline。

#### Scenario: 报告 majority-class baseline
- **WHEN** 用户评估 LiDAR-only baseline
- **THEN** 评估报告 MUST 包含每个预测 horizon 的 majority-class Top-1 baseline
- **AND** 评估报告 MUST 包含 LiDAR 模型每个 horizon 的 Top-1/Top-3 指标
- **AND** 报告 MUST 标明 LiDAR 模型是否超过 majority-class baseline

#### Scenario: 报告 last-beam baseline
- **WHEN** batch 或 dataset metadata 中可获得历史 beam label
- **THEN** 评估报告 MUST 包含 last-beam Top-1 和 Top-3 baseline
- **AND** 报告 MUST 标明 LiDAR 模型相对 last-beam baseline 的差距

### Requirement: LiDAR canonical 模型配置使用 modular BEV encoder
LiDAR strong/lightweight/supervised canonical 配置 MUST 使用修复后的 LiDAR BEV profile 和 `modular_sequence` + `lidar_cnn` encoder，并保持现有 logits/loss 输出契约不变。

#### Scenario: 构建 LiDAR teacher baseline
- **WHEN** 用户加载默认 LiDAR teacher/no-KD 配置
- **THEN** 系统 MUST 构建 `modular_sequence` LiDAR 模型
- **AND** 模型输入 MUST 是经过 baseline profile 处理的 `[B, T, C, H, W]` LiDAR BEV 张量
- **AND** 模型输出 MUST 继续兼容现有 `[B, T, num_classes]` logits 选择和 loss 计算路径

#### Scenario: LiDAR 模型不改变 future-only 对齐
- **WHEN** LiDAR 模型输出序列长度大于 `num_pred`
- **THEN** 系统 MUST 继续只使用最后 `num_pred` 个输出时隙对齐 `[t+1, t+2, t+3]` 标签
- **AND** 系统 MUST 不把历史窗口最后一个 beam 重新纳入训练 label

### Requirement: LiDAR legacy model names are removed
LiDAR legacy whole-model 注册名和 feature extractor `MODELS` 注册名 MUST 被 removed guard 拒绝。Current LiDAR canonical 配置 MUST 继续使用 `modular_sequence + lidar_cnn`。

#### Scenario: 请求 LiDAR legacy 注册名
- **WHEN** 用户请求 `lidar_teacher`、`lidar_student`、`lidar_strong`、`lidar_lightweight` 或 `lidar_feature_extractor`
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence + lidar_cnn + single_gru`

#### Scenario: LiDAR canonical 配置仍使用 modular path
- **WHEN** 用户加载 `configs/lidar/strong.yaml`、`configs/lidar/lightweight.yaml` 或 `configs/lidar/supervised.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** `model.primary.encoders.lidar.type` MUST 为 `lidar_cnn`

### Requirement: LiDAR 注册错误可诊断
LiDAR 相关注册错误 MUST 使用现有注册表错误风格，并在未知名称、重复名称或缺失必需参数时提供清晰错误信息。

#### Scenario: 请求未知 LiDAR 组件
- **WHEN** 配置中引用未注册的 LiDAR 模型或预处理器名称
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含请求的组件名称、注册表名称和可用组件列表

#### Scenario: LiDAR 构建参数缺失
- **WHEN** 配置中引用已注册 LiDAR 组件但缺少必需构造参数
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含缺失字段或原始构建错误

### Requirement: LiDAR 配置驱动实验
项目 MUST 支持通过配置文件启动 LiDAR-only 训练和评估。LiDAR-only 配置 MUST 使用当前 LiDAR dataset、preprocessing/cache contract、`model.primary`、统一训练/验证/评估、loss、optimizer、scheduler、checkpoint 和指标流程运行。

#### Scenario: 使用配置启动 LiDAR-only 训练
- **WHEN** 用户通过当前 CLI 传入 LiDAR-only 训练配置
- **THEN** 系统 MUST 构建包含 LiDAR 输入的 dataset、配置指定的 LiDAR primary model、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求 image、radar、GPS、teacher checkpoint 或 distiller
- **AND** LiDAR 输入 MUST 使用当前配置声明的 BEV、streaming stats 或 raw point cloud profile

#### Scenario: 使用配置启动 LiDAR-only 评估
- **WHEN** 用户通过当前 CLI 传入 LiDAR-only 评估配置和 LiDAR 模型权重
- **THEN** 系统 MUST 构建配置指定的 LiDAR primary model 并只使用 LiDAR 输入完成评估
- **AND** 系统 MUST 保存当前 metric profile 声明的 Top-K、DBA、loss 或诊断指标

### Requirement: LiDAR fusion 配置驱动实验
项目 MUST 支持通过 fusion `modalities` 配置启用 LiDAR。包含 LiDAR 的 fusion 配置 MUST 复用统一 fusion 训练和评估流程，并 MUST 构建单个 fusion primary model。

#### Scenario: 使用配置启动 image+radar+gps+lidar fusion 训练
- **WHEN** 用户通过训练入口传入 `modalities: ["image", "radar", "gps", "lidar"]` 的 fusion 配置
- **THEN** 系统 MUST 构建四个模态输入所需的 dataset 字段和 fusion primary model
- **AND** 系统 MUST 在 batch 准备阶段构造 image、radar、gps 和 lidar 输入

#### Scenario: 使用配置启动 LiDAR 参与的双模态 fusion 训练
- **WHEN** 用户通过训练入口传入包含 `lidar` 的任意合法双模态 fusion 配置
- **THEN** 系统 MUST 只准备 `modalities` 中列出的模态输入
- **AND** 未启用的模态字段 MUST 不影响训练启动

### Requirement: LiDAR 默认实验配置
项目 MUST 提供 LiDAR-only strong、lightweight、supervised 和包含 LiDAR 的 fusion 示例配置。所有默认 LiDAR primary 配置 MUST 使用当前 modular BEV encoder 默认参数。

#### Scenario: LiDAR 默认配置可构建
- **WHEN** 开发者加载 `configs/lidar/*.yaml`
- **THEN** 系统 MUST 能构建对应 dataset、model、loss、optimizer 和 scheduler
- **AND** 配置 MUST 使用 `model.primary`

#### Scenario: LiDAR fusion 示例配置可构建
- **WHEN** 开发者加载包含 LiDAR 的 `configs/fusion/*.yaml`
- **THEN** 系统 MUST 能构建对应 fusion primary model
- **AND** fusion `modalities` MUST 只包含合法模态名称

### Requirement: LiDAR dry-run 训练
项目 MUST 提供可在小数据或 fixture 上运行的 LiDAR smoke test 路径，用于验证 LiDAR forward、loss、backward、validation 和 checkpoint 保存。

#### Scenario: LiDAR dry-run 训练
- **WHEN** 开发者使用 synthetic、fixture 或小比例数据运行一次 LiDAR 短训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存的核心路径
