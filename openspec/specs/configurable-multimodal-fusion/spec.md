# configurable-multimodal-fusion Specification

## Purpose
TBD - created by archiving change add-gps-modality-fusion. Update Purpose after archive.
## Requirements
### Requirement: Fusion 模态选择配置
Fusion teacher 和 fusion student MUST 支持通过 `modalities` 配置选择参与融合的模态。`modalities` MUST 是 `image`、`radar`、`gps`、`lidar`、`mmwave` 的非空列表；默认值 MUST 保持既有 image+radar 行为。

#### Scenario: 默认 fusion 模态
- **WHEN** 用户构建 fusion 模型且未显式配置 `modalities`
- **THEN** 系统 MUST 使用 `["image", "radar"]`
- **AND** 系统 MUST 保持旧 image+radar 配置的模型输入和输出行为兼容

#### Scenario: 配置全部模态
- **WHEN** 用户配置 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** fusion 模型 MUST 创建 image、radar、gps、lidar 和 mmWave 五个分支
- **AND** fusion projection 的输入维度 MUST 与五个分支输出拼接维度一致

#### Scenario: 配置任意双模态
- **WHEN** 用户配置 `modalities` 为 `["image", "mmwave"]`、`["radar", "mmwave"]`、`["lidar", "mmwave"]` 或其它合法双模态组合
- **THEN** fusion 模型 MUST 只创建被启用模态的分支
- **AND** forward MUST 只要求被启用模态对应的输入张量

#### Scenario: 配置单模态 fusion
- **WHEN** 用户配置 `modalities` 为 `["image"]`、`["radar"]`、`["gps"]`、`["lidar"]` 或 `["mmwave"]`
- **THEN** fusion 模型 MUST 能构建并运行
- **AND** fusion projection MUST 只接收该单模态分支输出

### Requirement: Fusion 模态配置校验
系统 MUST 对 fusion `modalities` 做显式校验。空列表、重复模态或未知模态 MUST 在模型构建时抛出清晰错误。

#### Scenario: 空模态列表
- **WHEN** 用户配置 `modalities: []`
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 指出至少需要一个模态

#### Scenario: 未知模态
- **WHEN** 用户配置 `modalities` 包含 `image`、`radar`、`gps`、`lidar`、`mmwave` 之外的名称
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 包含非法模态名称

#### Scenario: 重复模态
- **WHEN** 用户配置 `modalities` 包含重复项
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 指出模态不能重复

### Requirement: Fusion teacher 支持 GPS
`fusion_teacher` MUST 能在启用 GPS 时融合 GPS 特征，并保持输出契约 `(pred, input_features, output_features)`。GPS 分支 MUST 使用与 GPS-only teacher 兼容的 feature extraction 风格。

#### Scenario: fusion_teacher 使用 GPS
- **WHEN** `fusion_teacher` 配置包含 `gps`
- **THEN** 模型 MUST 接收 GPS-Rel-Polar 输入张量 `[B, T, 3]`
- **AND** 模型 MUST 将 GPS 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_teacher 缺少启用模态输入
- **WHEN** `fusion_teacher` 配置包含 `gps` 但 forward 未收到 GPS 输入
- **THEN** 系统 MUST 抛出清晰错误

### Requirement: Fusion student 支持 GPS
`fusion_student` MUST 能在启用 GPS 时融合 GPS 特征，并保持 lightweight student 语义。GPS student 分支 MUST 使用轻量 MLP 或投影层，且默认 output hidden size MUST 与 teacher 对齐以支持 RKD。

#### Scenario: fusion_student 使用 GPS
- **WHEN** `fusion_student` 配置包含 `gps`
- **THEN** 模型 MUST 接收 GPS-Rel-Polar 输入张量 `[B, T, 3]`
- **AND** 模型 MUST 将 GPS 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_student KD 兼容
- **WHEN** fusion KD 配置中的 teacher 和 student 使用相同的 `modalities`
- **THEN** 系统 MUST 能完成 teacher/student forward
- **AND** logits KD 与 RKD MUST 能接收 fusion teacher/student 的 logits、input_features 和 output_features

### Requirement: Fusion 输入准备遵循模态选择
训练、验证和评估流程在 `experiment.task: fusion` 下 MUST 根据配置的 `modalities` 准备输入。未启用的模态 MUST 不被要求存在于 batch 中。

#### Scenario: fusion 只启用 image 和 gps
- **WHEN** fusion 配置的 `modalities` 为 `["image", "gps"]`
- **THEN** batch 准备 MUST 构造 image 和 gps 输入
- **AND** batch 准备 MUST 不要求 `radar_ra`、`radar_da`、`lidar` 或 `mmwave`

#### Scenario: fusion 启用全部模态
- **WHEN** fusion 配置的 `modalities` 为 `["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** batch 准备 MUST 构造 image、radar、gps、lidar 和 mmWave 输入
- **AND** 五个输入的 batch 和 sequence 维度 MUST 对齐

#### Scenario: fusion 只启用 LiDAR
- **WHEN** fusion 配置的 `modalities` 为 `["lidar"]`
- **THEN** batch 准备 MUST 构造 LiDAR 输入
- **AND** batch 准备 MUST 不要求 image、radar、gps 或 mmWave 字段

#### Scenario: fusion 只启用 mmWave
- **WHEN** fusion 配置的 `modalities` 为 `["mmwave"]`
- **THEN** batch 准备 MUST 构造 mmWave 输入
- **AND** batch 准备 MUST 不要求 image、radar、GPS 或 LiDAR 字段

### Requirement: Fusion teacher 支持 LiDAR
`fusion_teacher` MUST 能在启用 LiDAR 时融合 LiDAR BEV 特征，并保持输出契约 `(pred, input_features, output_features)`。LiDAR 分支 MUST 使用与 LiDAR-only teacher 兼容的 feature extraction 风格。

#### Scenario: fusion_teacher 使用 LiDAR
- **WHEN** `fusion_teacher` 配置包含 `lidar`
- **THEN** 模型 MUST 接收 LiDAR BEV 输入张量 `[B, T, C, H, W]`
- **AND** 模型 MUST 将 LiDAR 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_teacher 缺少 LiDAR 输入
- **WHEN** `fusion_teacher` 配置包含 `lidar` 但 forward 未收到 LiDAR 输入
- **THEN** 系统 MUST 抛出清晰错误

### Requirement: Fusion student 支持 LiDAR
`fusion_student` MUST 能在启用 LiDAR 时融合 LiDAR BEV 特征，并保持 lightweight student 语义。LiDAR student 分支 MUST 使用轻量 CNN 或 depthwise separable convolution，并通过 adaptive pooling 生成固定长度帧级 embedding。

#### Scenario: fusion_student 使用 LiDAR
- **WHEN** `fusion_student` 配置包含 `lidar`
- **THEN** 模型 MUST 接收 LiDAR BEV 输入张量 `[B, T, C, H, W]`
- **AND** 模型 MUST 将 LiDAR 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_student LiDAR KD 兼容
- **WHEN** fusion KD 配置中的 teacher 和 student 使用包含 LiDAR 的相同 `modalities`
- **THEN** 系统 MUST 能完成 teacher/student forward
- **AND** logits KD 与 RKD MUST 能接收 fusion teacher/student 的 logits、input_features 和 output_features

### Requirement: Fusion canonical 多模态配置矩阵
项目 MUST 为 `image`、`radar`、`gps`、`lidar`、`mmwave` 的所有必要多模态组合提供 canonical fusion 配置矩阵。多模态组合 MUST 覆盖全部 10 个双模态组合、10 个三模态组合、5 个四模态组合和 1 个五模态组合。每个组合 MUST 提供 teacher no-KD、student no-KD、logits KD 和 RKD 配置。

#### Scenario: 双模态 fusion 组合完整
- **WHEN** 开发者查看 `configs/fusion/`
- **THEN** 系统 MUST 提供 `image_radar`、`image_gps`、`image_lidar`、`image_mmwave`、`radar_gps`、`radar_lidar`、`radar_mmwave`、`gps_lidar`、`gps_mmwave` 和 `lidar_mmwave` 十个双模态 slug 的 canonical 配置
- **AND** 每个 slug MUST 具备 `<slug>_teacher_no_kd.yaml`、`<slug>_student_no_kd.yaml`、`<slug>_logits_kd.yaml` 和 `<slug>_rkd.yaml`

#### Scenario: 三模态 fusion 组合完整
- **WHEN** 开发者查看 `configs/fusion/`
- **THEN** 系统 MUST 提供 `image_radar_gps`、`image_radar_lidar`、`image_radar_mmwave`、`image_gps_lidar`、`image_gps_mmwave`、`image_lidar_mmwave`、`radar_gps_lidar`、`radar_gps_mmwave`、`radar_lidar_mmwave` 和 `gps_lidar_mmwave` 十个三模态 slug 的 canonical 配置
- **AND** 每个 slug MUST 具备 `<slug>_teacher_no_kd.yaml`、`<slug>_student_no_kd.yaml`、`<slug>_logits_kd.yaml` 和 `<slug>_rkd.yaml`

#### Scenario: 四模态 fusion 组合完整
- **WHEN** 开发者查看 `configs/fusion/`
- **THEN** 系统 MUST 提供 `image_radar_gps_lidar`、`image_radar_gps_mmwave`、`image_radar_lidar_mmwave`、`image_gps_lidar_mmwave` 和 `radar_gps_lidar_mmwave` 五个四模态 slug 的 canonical 配置
- **AND** 每个 slug MUST 具备 `<slug>_teacher_no_kd.yaml`、`<slug>_student_no_kd.yaml`、`<slug>_logits_kd.yaml` 和 `<slug>_rkd.yaml`

#### Scenario: 五模态 fusion 组合完整
- **WHEN** 开发者查看 `configs/fusion/`
- **THEN** 系统 MUST 提供 `image_radar_gps_lidar_mmwave_teacher_no_kd.yaml`、`image_radar_gps_lidar_mmwave_student_no_kd.yaml`、`image_radar_gps_lidar_mmwave_logits_kd.yaml` 和 `image_radar_gps_lidar_mmwave_rkd.yaml`

#### Scenario: 不重复提供 fusion 单模态入口
- **WHEN** 用户需要运行单模态 image、radar、GPS、LiDAR 或 mmWave 实验
- **THEN** 文档 MUST 引导用户使用 `configs/<modality>/` 下的单模态 canonical 配置
- **AND** fusion canonical 矩阵 MUST 不要求提供单模态 fusion duplicate 配置

### Requirement: Fusion canonical 配置语义
每个 canonical fusion 配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar`、`mmwave` 生成 slug，并 MUST 让 teacher 和 student 使用相同的 `modalities`。同一 slug 的四种配置 MUST 只改变训练角色和 KD 模式，不得改变模态集合。

#### Scenario: fusion teacher no-KD 配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_teacher_no_kd.yaml`
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 将被训练主模型配置为 `fusion_teacher`
- **AND** `model.teacher.modalities` 与 `model.student.modalities` MUST 等于 slug 表示的模态集合

#### Scenario: fusion student no-KD 配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_student_no_kd.yaml`
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 将被训练主模型配置为 `fusion_student`
- **AND** `model.teacher.modalities` 与 `model.student.modalities` MUST 等于 slug 表示的模态集合

#### Scenario: fusion logits KD 配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_logits_kd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: logits_kd`
- **AND** 配置 MUST 构建 frozen `fusion_teacher`
- **AND** 配置 MUST 构建可训练 `fusion_student`
- **AND** teacher 和 student 的 `modalities` MUST 相同
- **AND** 配置 MUST 默认解析同 slug 的 canonical teacher no-KD 输出中的 `best.pth`

#### Scenario: fusion RKD 配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_rkd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: rkd`
- **AND** 配置 MUST 构建 frozen `fusion_teacher`
- **AND** 配置 MUST 构建可训练 `fusion_student`
- **AND** teacher 和 student 的 `modalities` MUST 相同
- **AND** 配置 MUST 提供 RKD 参数并默认解析同 slug 的 canonical teacher no-KD 输出中的 `best.pth`

### Requirement: Fusion canonical 数据字段
canonical fusion 配置 MUST 根据 `modalities` 启用对应 dataset 字段，并不得要求未启用模态的数据列。启用 GPS 的配置 MUST 使用 GPS-Rel-Polar；启用 LiDAR 的配置 MUST 使用 LiDAR BEV 默认字段，并 MUST 沿用 LiDAR 懒加载和内存有界归一化语义；启用 mmWave 的配置 MUST 使用 64 维 dB receive-power 特征，并 MUST 复用训练集 mmWave scaler。

#### Scenario: 启用 GPS 的 fusion 配置
- **WHEN** canonical fusion 配置的 `modalities` 包含 `gps`
- **THEN** 配置 MUST 设置 `data.dataset.use_gps: true`
- **AND** 配置 MUST 设置 `gps_feature_mode: relative_polar`
- **AND** teacher 和 student 的 `gps_input_size` MUST 为 3

#### Scenario: 启用 LiDAR 的 fusion 配置
- **WHEN** canonical fusion 配置的 `modalities` 包含 `lidar`
- **THEN** 配置 MUST 设置 `data.dataset.use_lidar: true`
- **AND** 配置 MUST 提供 LiDAR BEV size、ROI 和归一化默认字段
- **AND** LiDAR 归一化默认字段 MUST 不要求 dataset 初始化阶段全量读取训练 split
- **AND** teacher 和 student MUST 使用与 LiDAR BEV 输入通道一致的 `lidar_channels`

#### Scenario: 启用 mmWave 的 fusion 配置
- **WHEN** canonical fusion 配置的 `modalities` 包含 `mmwave`
- **THEN** 配置 MUST 设置 `data.dataset.use_mmwave: true`
- **AND** 配置 MUST 设置 `mmwave_normalize: true`
- **AND** teacher 和 student 的 `mmwave_input_size` MUST 为 64

#### Scenario: fusion LiDAR streaming stats 显式启用
- **WHEN** canonical fusion 配置的 `modalities` 包含 `lidar` 且用户显式启用 LiDAR streaming stats
- **THEN** fusion dataloader MUST 使用与 LiDAR-only 配置相同的流式 stats 计算或 stats 文件复用逻辑
- **AND** 系统 MUST 不为 fusion 入口恢复全量 BEV concatenate 行为

#### Scenario: fusion mmWave scaler 复用
- **WHEN** canonical fusion 配置的 `modalities` 包含 `mmwave`
- **THEN** fusion dataloader MUST 使用与 mmWave-only 配置相同的训练集 scaler fit、保存和测试集复用逻辑
- **AND** 系统 MUST 不在测试 split 上重新 fit mmWave scaler

#### Scenario: 未启用模态不强制要求数据字段
- **WHEN** canonical fusion 配置的 `modalities` 不包含某个模态
- **THEN** 训练、验证和评估的 batch 准备 MUST 不要求该模态对应输入存在
- **AND** 模型 forward MUST 只接收启用模态对应的张量

### Requirement: Fusion legacy 入口兼容
项目 MUST 保留现有 fusion 示例和 legacy 入口作为兼容配置，并 MUST 在文档中说明它们对应的 canonical 配置。legacy 入口不得阻止 canonical 矩阵使用统一命名。

#### Scenario: image+radar legacy fusion 入口
- **WHEN** 用户运行 `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml` 或 `configs/fusion/rkd.yaml`
- **THEN** 系统 MUST 继续按 image+radar fusion 语义运行
- **AND** 文档 MUST 引导新实验优先使用 `image_radar_*` canonical 配置

#### Scenario: 既有 fusion 示例入口
- **WHEN** 用户运行现有 `image_gps_no_kd.yaml`、`radar_gps_no_kd.yaml`、`radar_lidar_no_kd.yaml` 或 all-modalities 示例配置
- **THEN** 系统 MUST 继续按其显式 `modalities` 语义运行
- **AND** 文档 MUST 说明对应的 canonical student no-KD 配置名称

### Requirement: Fusion teacher 支持 mmWave
`fusion_teacher` MUST 能在启用 mmWave 时融合 mmWave 64 维 receive-power 特征，并保持输出契约 `(pred, input_features, output_features)`。mmWave 分支 MUST 使用与 mmWave-only teacher 兼容的 feature extraction 风格。

#### Scenario: fusion_teacher 使用 mmWave
- **WHEN** `fusion_teacher` 配置包含 `mmwave`
- **THEN** 模型 MUST 接收 mmWave 输入张量 `[B, T, 64]`
- **AND** 模型 MUST 将 mmWave 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_teacher 缺少 mmWave 输入
- **WHEN** `fusion_teacher` 配置包含 `mmwave` 但 forward 未收到 mmWave 输入
- **THEN** 系统 MUST 抛出清晰错误

### Requirement: Fusion student 支持 mmWave
`fusion_student` MUST 能在启用 mmWave 时融合 mmWave 64 维 receive-power 特征，并保持 lightweight student 语义。mmWave student 分支 MUST 使用轻量 MLP 或投影层，且默认 output hidden size MUST 与 teacher 对齐以支持 RKD。

#### Scenario: fusion_student 使用 mmWave
- **WHEN** `fusion_student` 配置包含 `mmwave`
- **THEN** 模型 MUST 接收 mmWave 输入张量 `[B, T, 64]`
- **AND** 模型 MUST 将 mmWave 分支 embedding 与其它启用模态 embedding 在 feature 维拼接
- **AND** 模型 MUST 返回 `[B, T, num_classes]` logits

#### Scenario: fusion_student mmWave KD 兼容
- **WHEN** fusion KD 配置中的 teacher 和 student 使用包含 mmWave 的相同 `modalities`
- **THEN** 系统 MUST 能完成 teacher/student forward
- **AND** logits KD 与 RKD MUST 能接收 fusion teacher/student 的 logits、input_features 和 output_features

