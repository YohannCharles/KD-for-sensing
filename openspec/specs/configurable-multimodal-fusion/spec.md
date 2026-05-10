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
项目 MUST 为 `image`、`radar`、`gps`、`lidar`、`mmwave` 的所有必要多模态组合提供 canonical fusion 配置矩阵。多模态组合 MUST 覆盖全部 10 个双模态组合、10 个三模态组合、5 个四模态组合和 1 个五模态组合。每个组合 MUST 提供可加载的 teacher no-KD、student no-KD、logits KD 和 RKD canonical 配置路径；这些 canonical 配置 MAY 由 loader 生成，不要求每个路径都有实体 YAML 文件。

#### Scenario: 双模态 fusion 组合完整
- **WHEN** 开发者加载 `configs/fusion/<slug>_<mode>.yaml`
- **THEN** 系统 MUST 为 `image_radar`、`image_gps`、`image_lidar`、`image_mmwave`、`radar_gps`、`radar_lidar`、`radar_mmwave`、`gps_lidar`、`gps_mmwave` 和 `lidar_mmwave` 十个双模态 slug 提供 canonical 配置
- **AND** 每个 slug MUST 具备可加载的 `<slug>_teacher_no_kd.yaml`、`<slug>_student_no_kd.yaml`、`<slug>_logits_kd.yaml` 和 `<slug>_rkd.yaml` 路径

#### Scenario: 三模态 fusion 组合完整
- **WHEN** 开发者加载 `configs/fusion/<slug>_<mode>.yaml`
- **THEN** 系统 MUST 为 `image_radar_gps`、`image_radar_lidar`、`image_radar_mmwave`、`image_gps_lidar`、`image_gps_mmwave`、`image_lidar_mmwave`、`radar_gps_lidar`、`radar_gps_mmwave`、`radar_lidar_mmwave` 和 `gps_lidar_mmwave` 十个三模态 slug 提供 canonical 配置
- **AND** 每个 slug MUST 具备可加载的 `<slug>_teacher_no_kd.yaml`、`<slug>_student_no_kd.yaml`、`<slug>_logits_kd.yaml` 和 `<slug>_rkd.yaml` 路径

#### Scenario: 四模态 fusion 组合完整
- **WHEN** 开发者加载 `configs/fusion/<slug>_<mode>.yaml`
- **THEN** 系统 MUST 为 `image_radar_gps_lidar`、`image_radar_gps_mmwave`、`image_radar_lidar_mmwave`、`image_gps_lidar_mmwave` 和 `radar_gps_lidar_mmwave` 五个四模态 slug 提供 canonical 配置
- **AND** 每个 slug MUST 具备可加载的 `<slug>_teacher_no_kd.yaml`、`<slug>_student_no_kd.yaml`、`<slug>_logits_kd.yaml` 和 `<slug>_rkd.yaml` 路径

#### Scenario: 五模态 fusion 组合完整
- **WHEN** 开发者加载五模态 fusion canonical 配置
- **THEN** 系统 MUST 提供可加载的 `image_radar_gps_lidar_mmwave_teacher_no_kd.yaml`、`image_radar_gps_lidar_mmwave_student_no_kd.yaml`、`image_radar_gps_lidar_mmwave_logits_kd.yaml` 和 `image_radar_gps_lidar_mmwave_rkd.yaml` 路径

#### Scenario: 不重复提供 fusion 单模态入口
- **WHEN** 用户需要运行单模态 image、radar、GPS、LiDAR 或 mmWave 实验
- **THEN** 文档 MUST 引导用户使用 `configs/<modality>/` 下的单模态 canonical 配置
- **AND** fusion canonical 矩阵 MUST 不要求提供单模态 fusion duplicate 配置

### Requirement: Fusion canonical 配置语义
每个 canonical fusion 配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar`、`mmwave` 生成 slug，并 MUST 让 teacher 和 student 使用相同的 `modalities`。同一 slug 的四种配置 MUST 只改变训练角色和 KD 模式，不得改变模态集合。canonical 配置语义 MUST 不依赖实体 YAML 文件是否存在。

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

### Requirement: Fusion teacher image 分支复用单模态特征提取器
`fusion_teacher` 在启用 image 模态时 MUST 使用 image-only teacher 暴露的 `ImageFeatureExtractor` 作为帧级特征提取器。系统 MUST 不再为 fusion teacher 维护单独的旧版 image feature extractor 副本。

#### Scenario: 构建包含 image 的 fusion teacher
- **WHEN** 用户构建 `fusion_teacher` 且 `modalities` 包含 `image`
- **THEN** 模型 MUST 将 `image_feature_extractor` 初始化为 `ImageFeatureExtractor`
- **AND** image 分支输出 MUST 保持 `[B, T, feature_size]` 形状以参与 fusion projection

#### Scenario: 构建不包含 image 的 fusion teacher
- **WHEN** 用户构建 `fusion_teacher` 且 `modalities` 不包含 `image`
- **THEN** 模型 MUST 不创建 image feature extractor
- **AND** 缺失 image 输入不得阻止该 fusion teacher forward

#### Scenario: 旧 fusion teacher checkpoint 结构不匹配
- **WHEN** 用户使用严格加载将旧 `FusionImageFeatureExtractor` 结构的 `fusion_teacher` checkpoint 加载到新模型
- **THEN** 系统 MUST 报告 checkpoint 结构不匹配
- **AND** 错误信息 MUST 包含 missing keys 或 unexpected keys 诊断

### Requirement: Fusion 配置选择 CRAF 模型
Fusion 配置 MUST 能显式选择 CRAF 或 CRAF baseline 模型，同时继续使用现有 `modalities` 字段描述参与融合的模态集合。

#### Scenario: 配置 CRAF fusion
- **WHEN** 用户在 fusion 配置中设置 `model.student.type: craf_fusion`
- **THEN** 系统 MUST 使用 `model.student.modalities` 构建 CRAF 模型
- **AND** 系统 MUST 继续使用 `experiment.task: fusion` 的 batch 输入准备流程

#### Scenario: 配置 token transformer fusion
- **WHEN** 用户在 fusion 配置中设置 token-only transformer baseline 的注册名
- **THEN** 系统 MUST 使用同一模态集合构建不带 reliability gate 的 token fusion baseline

#### Scenario: legacy fusion 配置不变
- **WHEN** 用户继续运行既有 `fusion_teacher` 或 `fusion_student` 配置
- **THEN** 系统 MUST 保持 early-concat fusion 行为
- **AND** 系统 MUST 不隐式启用 CRAF 训练 loss 或 diagnostics

### Requirement: CRAF 配置复用 fusion 模态校验
CRAF 和 CRAF baseline 配置 MUST 复用现有 fusion 模态标准化和校验语义。模态顺序、未知模态错误、重复模态错误和未启用模态输入行为 MUST 与当前 fusion 配置一致。

#### Scenario: 乱序模态标准化
- **WHEN** 用户在 CRAF 配置中设置 `modalities: ["lidar", "image", "gps"]`
- **THEN** 系统 MUST 将模态顺序标准化为项目固定顺序
- **AND** reliability 输出、token 输出和日志 MUST 使用标准化后的顺序

#### Scenario: 未启用模态不读取输入
- **WHEN** CRAF 配置不包含 `mmwave`
- **THEN** batch 准备和模型 forward MUST 不要求 `mmwave` 输入

### Requirement: CRAF canonical 与示例配置
项目 MUST 提供可运行的 CRAF 示例配置和 baseline 示例配置，用于当前项目数据和训练入口的 smoke test 与实验对比。

#### Scenario: all-modalities CRAF 配置
- **WHEN** 用户加载 all-modalities CRAF 示例配置
- **THEN** 配置 MUST 启用 image、radar、GPS、LiDAR 和 mmWave
- **AND** 配置 MUST 设置 CRAF 所需的模型、loss、counterfactual 和输出字段

#### Scenario: image-radar CRAF 配置
- **WHEN** 用户加载 image+radar CRAF 示例配置
- **THEN** 配置 MUST 使用与现有 image+radar fusion 可比较的 dataset split、num classes、seq length 和 num pred

#### Scenario: baseline 配置可加载
- **WHEN** 用户加载 token transformer 或 early concat transformer baseline 配置
- **THEN** 配置 MUST 通过现有 config loader 构建成功
- **AND** 配置 MUST 不要求新增训练入口

### Requirement: CRAF 稳定化配置字段
CRAF 示例配置 MUST 能表达稳定化训练策略，包括 warmup、CE-only counterfactual、ignore band、softmax gate、temperature schedule 和 auxiliary loss schedule。

#### Scenario: all-modalities 稳定化 CRAF 配置
- **WHEN** 用户加载 all-modalities CRAF 稳定化配置
- **THEN** 配置 MUST 启用五个标准模态
- **AND** 配置 MUST 设置 warmup epoch、counterfactual 起始 epoch、`context_marginal` 或等价反事实模式、CE-only delta 和 ignore band

#### Scenario: softmax gate 配置
- **WHEN** 用户在 CRAF 配置中设置 `model.student.reliability.gate_type: softmax`
- **THEN** 配置 MUST 能传递 gate temperature、temperature schedule 和 `min_gate`
- **AND** 模型构建 MUST 不影响配置为 sigmoid gate 的旧实验

#### Scenario: auxiliary 与 beam soft schedule 配置
- **WHEN** 用户配置 CRAF 附加 loss
- **THEN** 配置 MUST 能表达 warmup-only 单模态 auxiliary loss 和 beam soft loss 权重
- **AND** 权重为 0 的附加 loss MUST 保持关闭

### Requirement: Teacher-prior CRAF 配置入口
Fusion 配置 MUST 支持 teacher-prior CRAF Stage 2、Stage 3 和消融实验入口。新增配置 MUST 使用现有 fusion 数据字段、固定模态顺序和 CRAF 显式模型类型，不得改变 legacy fusion 默认行为。

#### Scenario: Stage 2 主实验配置可加载
- **WHEN** 用户加载 Stage 2 teacher-init prior CRAF 配置
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 使用 `model.student.type: craf_fusion`
- **AND** 配置 MUST 启用 `["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** 配置 MUST 提供 teacher registry 路径、`gate_type: prior_residual_sigmoid`、prior regularization 权重和 frozen encoder 策略

#### Scenario: Stage 3 主实验配置可加载
- **WHEN** 用户加载 Stage 3 selective fine-tuning CRAF 配置
- **THEN** 配置 MUST 设置 Stage 2 checkpoint 加载路径
- **AND** 配置 MUST 提供 `finetune.unfreeze_modalities` 和 `finetune.freeze_modalities`
- **AND** 配置 MUST 提供 Stage 3 参数组学习率

#### Scenario: Teacher-prior 消融配置可加载
- **WHEN** 用户加载 teacher-init no-prior、prior random-encoder、teacher-init fixed-prior 或 teacher-init prior-residual 消融配置
- **THEN** 每个配置 MUST 明确 `teacher.load_encoders` 和 gate 类型
- **AND** 每个配置 MUST 使用同一场景、同一 split、同一模态集合和同一基础训练超参数，除消融目标字段外不得隐式改变实验条件

### Requirement: CRAF gate 类型配置
`craf_fusion` 配置 MUST 支持 `none`、`fixed_prior`、`prior_residual_sigmoid` 和旧 gate 类型。每种 gate 类型 MUST 有清晰的 mask 语义和 diagnostics 行为。

#### Scenario: gate none
- **WHEN** 配置设置 `gate_type: none`
- **THEN** CRAF MUST 对所有可用模态使用 gate 1
- **AND** 不可用模态 MUST 仍被 mask 排除

#### Scenario: gate fixed prior
- **WHEN** 配置设置 `gate_type: fixed_prior`
- **THEN** CRAF MUST 使用配置或 teacher registry 中的 prior 作为 gate
- **AND** gate MUST 不引入可学习 residual

#### Scenario: gate prior residual sigmoid
- **WHEN** 配置设置 `gate_type: prior_residual_sigmoid`
- **THEN** CRAF MUST 构建 PriorResidualGate
- **AND** 配置 MUST 能控制 `min_gate`、hidden dim、confidence feature 和 residual 零初始化

### Requirement: Legacy fusion 配置不变
新增 teacher-prior CRAF 配置 MUST 不改变既有 fusion、token transformer、CRAF stabilized 和 fixed prior sanity 配置的解析结果。

#### Scenario: legacy fusion 仍按 image+radar 运行
- **WHEN** 用户加载既有 `configs/fusion/no_kd.yaml` 或 canonical image+radar fusion 配置
- **THEN** 配置 MUST 继续构建 legacy fusion teacher/student
- **AND** 配置 MUST 不自动启用 teacher registry、prior residual gate 或 selective finetune

#### Scenario: 已有 fixed prior sanity 保持语义
- **WHEN** 用户加载 `craf_all_modalities_fixed_prior_sanity` 配置
- **THEN** 配置 MUST 继续使用 fixed prior gate
- **AND** 配置 MUST 不自动加载 teacher encoder

### Requirement: Fusion G2D 五模态配置入口
项目 MUST 提供五模态 G2D fusion 配置入口。配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar`、`mmwave`，MUST 设置 `experiment.task: fusion`，MUST 设置 `distillation.type: g2d`，并 MUST 保持 `model.num_pred: 3`。

#### Scenario: 加载 G2D-lite 配置
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml`
- **THEN** 配置 MUST 启用五个 fusion student 输入模态
- **AND** 配置 MUST 设置 `distillation.type: g2d`
- **AND** 配置 MUST 设置 G2D mode 为 `lite`

#### Scenario: 加载 G2D-global 配置
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml`
- **THEN** 配置 MUST 启用五个 fusion student 输入模态
- **AND** 配置 MUST 设置 `distillation.type: g2d`
- **AND** 配置 MUST 设置 G2D mode 为 `global`
- **AND** 配置 MUST 启用 SMP

#### Scenario: 加载 G2D-horizon 配置
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml`
- **THEN** 配置 MUST 启用五个 fusion student 输入模态
- **AND** 配置 MUST 设置 `distillation.type: g2d`
- **AND** 配置 MUST 设置 G2D mode 为 `horizon_diagnostic`

### Requirement: Fusion student exposes modality features for G2D
fusion student MUST 在不改变主 logits 契约的前提下，为 G2D feature KD 暴露 per-modality branch features。该输出 MUST 能通过 `adapt_model_output()` 的 diagnostics 传递给 G2D distiller。

#### Scenario: legacy fusion_student 输出 modality features
- **WHEN** `fusion_student` 前向完成且启用了多个模态
- **THEN** 输出 diagnostics MUST 包含按模态命名的 branch feature
- **AND** 每个 branch feature MUST 保持 batch 和 sequence 维度与 logits 对齐
- **AND** 主 logits MUST 继续能被解析为 `[B,T,C]`

#### Scenario: CRAF 或 MARF 输出 token features
- **WHEN** G2D student 使用 CRAF、MARF 或 token transformer 风格 fusion 模型
- **THEN** G2D feature extractor MUST 能从 `token_features` 和 `modalities` diagnostics 中按模态拆分 feature
- **AND** 拆分后的 feature MUST 能参与 feature KD

