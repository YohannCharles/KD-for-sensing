## MODIFIED Requirements

### Requirement: 跨模态可比较 split 配置
项目 MUST 提供可用于单模态和多模态横向比较的统一 split 配置方式。默认和 canonical 实验配置 MUST 让 image、radar、GPS、LiDAR、mmWave 和 fusion 实验引用同一组 train/test CSV。默认统一 CSV 文件名 MUST 继续可配置为 `train_seqs_RA_GPS_LIDAR.csv` 和 `test_seqs_RA_GPS_LIDAR.csv`；当启用 mmWave 时，这组 CSV MUST 由预处理流程生成可选的 `mmwave1..mmwaveN` 列。

#### Scenario: 使用统一 split 运行单模态实验
- **WHEN** 用户将 image、radar、GPS、LiDAR 和 mmWave 单模态配置指向同一组 train/test CSV
- **THEN** 系统 MUST 使用相同样本集合构建各模态 dataset
- **AND** 训练或评估输出 MUST 记录相同的 CSV 路径和样本数

#### Scenario: 使用统一 split 运行 fusion 实验
- **WHEN** 用户将 fusion 配置指向与单模态相同的 train/test CSV
- **THEN** 系统 MUST 使用相同样本集合构建 fusion dataset
- **AND** 未启用模态不得影响该 split 的可用性

#### Scenario: 默认配置使用统一 split
- **WHEN** 开发者查看默认 image、radar、GPS、LiDAR、mmWave 和 fusion 实验配置
- **THEN** 每个配置 MUST 指向同一组 train/test CSV，默认文件名为 `train_seqs_RA_GPS_LIDAR.csv` 和 `test_seqs_RA_GPS_LIDAR.csv`
- **AND** 输出 MUST 清晰记录该统一 split 的路径和样本数

#### Scenario: mmWave 统一 split 包含输入列
- **WHEN** 用户运行默认 mmWave 或包含 mmWave 的 fusion 配置
- **THEN** 该配置引用的统一 split CSV MUST 包含 `mmwave1..mmwaveN` 列
- **AND** 如果列缺失，系统 MUST 抛出清晰错误并提示重新运行启用 mmWave 的序列预处理

## ADDED Requirements

### Requirement: mmWave 配置驱动实验
项目 MUST 支持通过配置文件启动 mmWave-only 训练和评估。mmWave-only 配置 MUST 使用 `experiment.task: mmwave`，并通过统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和指标流程运行。

#### Scenario: 使用配置启动 mmWave-only 训练
- **WHEN** 用户通过新 CLI 传入 mmWave-only 训练配置
- **THEN** 系统 MUST 构建包含 mmWave 输入的 dataset、配置指定的 mmWave teacher/student 模型、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求图像、雷达、GPS 或 LiDAR 输入
- **AND** mmWave 输入 MUST 使用 `[B, T, 64]` 的 dB receive-power 特征序列

#### Scenario: 使用配置启动 mmWave-only 评估
- **WHEN** 用户通过新 CLI 传入 mmWave-only 评估配置和 mmWave 模型权重
- **THEN** 系统 MUST 构建配置指定的 mmWave 模型并只使用 mmWave 输入完成评估
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标
- **AND** 评估流程 MUST 复用训练时保存的 mmWave scaler

### Requirement: mmWave fusion 配置驱动实验
项目 MUST 支持通过 fusion `modalities` 配置启用 mmWave。包含 mmWave 的 fusion 配置 MUST 复用统一 fusion 训练和评估流程。

#### Scenario: 使用配置启动五模态 fusion 训练
- **WHEN** 用户通过训练入口传入 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]` 的 fusion 配置
- **THEN** 系统 MUST 构建五个模态输入所需的 dataset 字段和 fusion teacher/student 模型
- **AND** 系统 MUST 在 batch 准备阶段构造 image、radar、gps、lidar 和 mmWave 输入

#### Scenario: 使用配置启动 mmWave 参与的双模态 fusion 训练
- **WHEN** 用户通过训练入口传入包含 `mmwave` 的任意合法双模态 fusion 配置
- **THEN** 系统 MUST 只准备 `modalities` 中列出的模态输入
- **AND** 未启用的模态字段 MUST 不影响训练启动

### Requirement: mmWave 默认实验配置
项目 MUST 提供 mmWave-only teacher no-KD、student no-KD、logits KD、RKD 配置和包含 mmWave 的 canonical fusion 配置。所有默认 mmWave teacher/student 配置 MUST 使用 `mmwave_input_size: 64`、`mmwave_normalize: true` 和 `gru_params: [64, 64, 1]`。

#### Scenario: mmWave 默认配置可构建
- **WHEN** 开发者加载 `configs/mmwave/*.yaml`
- **THEN** 系统 MUST 能构建对应 dataset、model、loss、distiller、optimizer 和 scheduler
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** teacher 和 student 配置的 `mmwave_input_size` MUST 为 64

#### Scenario: mmWave KD 配置默认 checkpoint 来源
- **WHEN** 用户运行 `configs/mmwave/logits_kd.yaml` 或 `configs/mmwave/rkd.yaml` 且未显式覆盖 teacher 权重
- **THEN** 系统 MUST 从 mmWave teacher no-KD 训练输出或最佳 checkpoint registry 解析 teacher checkpoint
- **AND** 该默认解析 MUST 与其它单模态 KD 配置的 checkpoint 优先级一致

### Requirement: mmWave 预处理入口
预处理入口 MUST 支持通过配置生成带 mmWave 输入列的 Scenario 9 sequence CSV。该入口 MUST 允许配置 mmWave 源列和 fallback 列，并保持未启用 mmWave 的序列生成行为兼容。

#### Scenario: 运行带 mmWave 列的序列生成
- **WHEN** 用户通过预处理入口启用 `include_mmwave: true`
- **THEN** 系统 MUST 在训练和测试序列 CSV 中写入历史 `mmwave1..mmwaveN` 路径列
- **AND** 输出 CSV MUST 可被启用 mmWave 的 Scenario 9 dataset 直接读取

#### Scenario: mmWave 源列缺失
- **WHEN** 用户启用 mmWave 序列列输出但原始 CSV 不包含配置的 mmWave 源列或 fallback 列
- **THEN** 系统 MUST 抛出包含源列名、fallback 列名和 CSV 路径的清晰错误
