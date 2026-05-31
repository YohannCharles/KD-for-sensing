# configurable-multimodal-fusion Specification

## Purpose
定义多模态 fusion 配置的模态选择、模型类型、KD/辅助目标、高级方法兼容以及 virtual overlay 接管实体 YAML 后的运行语义。
## Requirements
### Requirement: Fusion 模态选择配置
Fusion teacher 和 fusion student MUST 支持通过 `modalities` 配置选择参与融合的模态。`modalities` MUST 是 `image`、`radar`、`gps`、`lidar`、`mmwave`、`csi` 的非空列表；默认值 MUST 保持既有 image+radar 行为。

#### Scenario: 默认 fusion 模态
- **WHEN** 用户构建 fusion 模型且未显式配置 `modalities`
- **THEN** 系统 MUST 使用 `["image", "radar"]`
- **AND** 系统 MUST 保持旧 image+radar 配置的模型输入和输出行为兼容

#### Scenario: 配置全部模态
- **WHEN** 用户配置 `modalities: ["image", "radar", "gps", "lidar", "mmwave", "csi"]`
- **THEN** fusion 模型 MUST 创建 image、radar、gps、lidar、mmWave 和 CSI 六个分支
- **AND** fusion projection 的输入维度 MUST 与六个分支输出拼接维度一致

#### Scenario: 配置任意双模态
- **WHEN** 用户配置 `modalities` 为 `["image", "csi"]`、`["radar", "csi"]`、`["mmwave", "csi"]` 或其它合法双模态组合
- **THEN** fusion 模型 MUST 只创建被启用模态的分支
- **AND** forward MUST 只要求被启用模态对应的输入张量

#### Scenario: 配置单模态 fusion
- **WHEN** 用户配置 `modalities` 为 `["image"]`、`["radar"]`、`["gps"]`、`["lidar"]`、`["mmwave"]` 或 `["csi"]`
- **THEN** fusion 模型 MUST 能构建并运行
- **AND** fusion projection MUST 只接收该单模态分支输出

### Requirement: Fusion 模态配置校验
系统 MUST 对 fusion `modalities` 做显式校验。空列表、重复模态或未知模态 MUST 在模型构建时抛出清晰错误。

#### Scenario: 空模态列表
- **WHEN** 用户配置 `modalities: []`
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 指出至少需要一个模态

#### Scenario: 未知模态
- **WHEN** 用户配置 `modalities` 包含 `image`、`radar`、`gps`、`lidar`、`mmwave`、`csi` 之外的名称
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
每个 canonical fusion 配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar`、`mmwave` 生成 slug，并 MUST 让 teacher 和 student 使用相同的 `modalities`。推荐/default fusion student 路线 MUST 使用 `cls_token_transformer_fusion` 作为混合方式；已退役的 CRAF、MARF、G2D 配置、overlay 和 alias MUST 不再作为可解析 canonical 或高级方法入口。同一 slug 的四种配置 MUST 只改变训练角色和 KD 模式，不得改变模态集合。canonical 配置语义 MUST 不依赖实体 YAML 文件是否存在。

#### Scenario: fusion teacher no-KD 配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_teacher_no_kd.yaml`
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 将被训练主模型配置为 `fusion_teacher` 或明确命名的 teacher baseline
- **AND** `model.teacher.modalities` 与 `model.student.modalities` MUST 等于 slug 表示的模态集合

#### Scenario: fusion student no-KD 默认配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_student_no_kd.yaml` 或推荐/default fusion no-KD student 配置
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 将被训练主模型配置为 `cls_token_transformer_fusion`
- **AND** `model.teacher.modalities` 与 `model.student.modalities` MUST 等于 slug 表示的模态集合

#### Scenario: fusion logits KD 默认配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_logits_kd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: logits_kd`
- **AND** 配置 MUST 构建 frozen `fusion_teacher` 或明确命名的 teacher baseline
- **AND** 配置 MUST 构建可训练 `cls_token_transformer_fusion`
- **AND** teacher 和 student 的 `modalities` MUST 相同
- **AND** 配置 MUST 默认解析同 slug 的 canonical teacher no-KD 输出中的 `best.pth`

#### Scenario: fusion RKD 默认配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_rkd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: rkd`
- **AND** 配置 MUST 构建 frozen `fusion_teacher` 或明确命名的 teacher baseline
- **AND** 配置 MUST 构建可训练 `cls_token_transformer_fusion`
- **AND** teacher 和 student 的 `modalities` MUST 相同
- **AND** 配置 MUST 提供 RKD 参数并默认解析同 slug 的 canonical teacher no-KD 输出中的 `best.pth`

#### Scenario: 退役高级方法配置不可解析
- **WHEN** 用户加载 CRAF、MARF 或 G2D 的实体 YAML、virtual alias 或 overlay recipe
- **THEN** 系统 MUST 拒绝该配置路径或方法名
- **AND** 系统 MUST 不生成等价配置或兼容重定向

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

### Requirement: Fusion 模型公开类名表达 teacher/student 职责
Early-concat fusion teacher 和 student MUST 暴露职责明确的公开 Python 类名。`fusion_teacher` 注册名 MUST 构建 `FusionTeacherModalityNet`，`fusion_student` 注册名 MUST 构建 `FusionStudentModalityNet`。旧类名 `old fusion teacher class alias` 和 `old fusion student class alias` MUST 不再作为兼容 alias 导出。

#### Scenario: 构建 fusion teacher 返回新类名
- **WHEN** 开发者通过 `MODELS.build()` 构建 `type: fusion_teacher`
- **THEN** 系统 MUST 返回 `FusionTeacherModalityNet` 实例
- **AND** 该实例 MUST 保持既有 `fusion_teacher` forward 输出契约

#### Scenario: 构建 fusion student 返回新类名
- **WHEN** 开发者通过 `MODELS.build()` 构建 `type: fusion_student`
- **THEN** 系统 MUST 返回 `FusionStudentModalityNet` 实例
- **AND** 该实例 MUST 保持既有 `fusion_student` forward 输出契约

#### Scenario: 旧类名 alias 被拒绝
- **WHEN** 开发者导入 `old fusion teacher class alias` 或 `old fusion student class alias`
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向 `FusionTeacherModalityNet` 或 `FusionStudentModalityNet`

### Requirement: 当前高级 fusion overlay 边界
当前保留的高级 fusion overlay MUST 只覆盖已批准的 objective-aware 或调试入口。已退役的 CRAF、MARF、G2D 和相关 ablation 配置 MUST 不再由 overlay recipe 或 virtual alias 生成。

#### Scenario: 保留实体 YAML 优先
- **WHEN** 用户加载一个仍存在的 `configs/fusion/*.yaml` 文件
- **THEN** 系统 MUST 使用该实体 YAML 的内容
- **AND** 不得用 virtual overlay 规则覆盖实体 YAML 中显式配置的字段

#### Scenario: 退役实体 YAML 不被 virtual 接管
- **WHEN** 被删除 YAML 属于 CRAF、MARF 或 G2D
- **THEN** 系统 MUST 将其视为不支持路径
- **AND** 系统 MUST 不为其提供 virtual fallback

#### Scenario: 配置矩阵测试覆盖 overlay
- **WHEN** 开发者运行 fusion 配置矩阵测试
- **THEN** 测试 MUST 覆盖当前保留 overlay 入口的可加载性和关键字段
- **AND** 测试 MUST 验证仍保留的实体 YAML 按兼容语义加载

### Requirement: Fusion 支持 RGB image profile
包含 image modality 的 fusion 配置 MUST 显式或隐式携带 image profile。默认 image profile MUST 为 `rgb_imagenet`；模块化 fusion 或 ResNet-18 fusion 配置 MUST 默认使用 `rgb_imagenet`，并让 dataset、batch 准备和 image encoder 使用同一个 profile。

#### Scenario: 模块化 fusion 使用 RGB image
- **WHEN** 用户运行模块化 fusion 配置且默认或设置 `image_profile: rgb_imagenet`
- **THEN** dataset MUST 返回 RGB/ImageNet 标准化 image tensor
- **AND** image encoder MUST 接收 3 通道 RGB 输入
- **AND** 其它启用模态的输入准备语义 MUST 保持不变

### Requirement: Fusion image encoder 与 profile 校验
Fusion 模型构建 MUST 校验启用 image modality 时的 image encoder 和 image profile 是否匹配。该校验 MUST 覆盖当前保留的 fusion、token transformer fusion 和模块化 fusion 入口，或在不支持某配置的入口处给出明确错误。

#### Scenario: fusion 使用 RGB profile
- **WHEN** 用户为 `fusion_teacher`、`fusion_student` 或 token transformer fusion 配置 `image_profile: rgb_imagenet`
- **THEN** 系统 MUST 构建或要求 3 通道 image branch
- **AND** 错误信息 MUST 在通道数不匹配时说明期望和实际通道数

#### Scenario: ResNet-18 fusion 使用 RGB profile
- **WHEN** 用户在 fusion 中选择 ResNet-18 image encoder 且 image profile 为 `rgb_imagenet`
- **THEN** 系统 MUST 构建或运行该配置
- **AND** image batch MUST 具有 3 通道 RGB/ImageNet 输入

#### Scenario: 已退役 fusion 方法不参与 profile 校验
- **WHEN** 配置请求 CRAF 或 MARF 风格 fusion
- **THEN** 系统 MUST 在 profile 校验前拒绝该模型类型
- **AND** 系统 MUST 不进入 CRAF/MARF 专属 image branch 构建逻辑

### Requirement: Modular fusion 复用现有模态选择语义
新的模块化 fusion 入口 MUST 复用现有 `modalities` 校验、固定模态顺序和 batch 输入字段语义。未启用的模态 MUST 不被 dataset、batch 准备、encoder 或 core 要求存在。

#### Scenario: 模块化 fusion 只启用 image 和 gps
- **WHEN** 模块化 fusion 配置的 `modalities` 为 `["image", "gps"]`
- **THEN** batch 准备 MUST 只构造 `image_batch` 和 `gps_batch`
- **AND** 模型 forward MUST 不要求 radar、LiDAR 或 mmWave 输入

#### Scenario: 模块化 fusion 启用全部模态
- **WHEN** 模块化 fusion 配置的 `modalities` 为 `["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** 系统 MUST 为五个模态构建 encoder 和 projector
- **AND** representation core 接收的模态顺序 MUST 遵循模态契约固定顺序

### Requirement: 默认 fusion no-KD 使用 CLS-token Transformer
推荐/default fusion no-KD 配置 MUST 使用 CLS-token Transformer fusion 作为混合方式。显式命名为 legacy 或 early-concat 的保留配置 MUST 保持其方法语义，不得被默认行为覆盖；已退役的 CRAF、MARF、G2D 或相关 ablation 配置 MUST 不再作为支持入口存在。

#### Scenario: 加载五模态默认 fusion no-KD
- **WHEN** 用户加载推荐的五模态 fusion no-KD 配置
- **THEN** 配置 MUST 启用 `["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** `model.student.type` MUST 为 `cls_token_transformer_fusion`
- **AND** 配置 MUST 设置 CLS-token Transformer 所需的 `d_model`、`num_heads`、`num_layers` 或等价默认值

#### Scenario: 加载双模态默认 fusion no-KD
- **WHEN** 用户加载推荐的双模态 fusion no-KD 配置
- **THEN** 配置 MUST 使用 slug 表示的两个模态
- **AND** `model.student.type` MUST 为 `cls_token_transformer_fusion`
- **AND** dataset 字段 MUST 只启用该组合需要的模态数据

#### Scenario: 显式 early-concat baseline 不被覆盖
- **WHEN** 用户加载显式 early-concat、legacy fusion 或模块化 `early_concat_gru` 配置
- **THEN** 系统 MUST 保持该配置声明的模型类型和 representation core
- **AND** 系统 MUST 不将其静默改写为 `cls_token_transformer_fusion`

#### Scenario: 已退役方法不被默认配置保留
- **WHEN** 用户查找默认或高级 fusion no-KD 推荐入口
- **THEN** 项目 MUST 不再提供 CRAF、MARF、G2D 或其 ablation 配置作为推荐入口

### Requirement: CLS-token Transformer 配置复用 fusion 数据字段
CLS-token Transformer fusion 配置 MUST 复用现有 fusion 数据字段和模态启用语义。启用 GPS、LiDAR 或 mmWave 时，配置 MUST 使用与其它 fusion 配置一致的数据字段、归一化和输入准备逻辑。

#### Scenario: 启用 GPS
- **WHEN** CLS-token Transformer fusion 配置的 `modalities` 包含 `gps`
- **THEN** 配置 MUST 设置 `data.dataset.use_gps: true`
- **AND** 配置 MUST 设置 `gps_feature_mode: relative_polar`
- **AND** `gps_input_size` MUST 为 3

#### Scenario: 启用 LiDAR
- **WHEN** CLS-token Transformer fusion 配置的 `modalities` 包含 `lidar`
- **THEN** 配置 MUST 设置 `data.dataset.use_lidar: true`
- **AND** 配置 MUST 沿用 LiDAR BEV 默认字段、缓存和内存有界归一化语义
- **AND** 模型 `lidar_channels` MUST 与 LiDAR BEV 输入通道一致

#### Scenario: 启用 mmWave
- **WHEN** CLS-token Transformer fusion 配置的 `modalities` 包含 `mmwave`
- **THEN** 配置 MUST 设置 `data.dataset.use_mmwave: true`
- **AND** 配置 MUST 设置 `mmwave_normalize: true`
- **AND** `mmwave_input_size` MUST 为 64

### Requirement: Fusion 多任务配置入口
Fusion 配置 MUST 能声明多任务辅助监督相关选项，包括启用状态、遮挡阈值分位数、位置目标来源、辅助 head 开关和 loss 权重。默认 fusion 配置 MUST 保持 beam-only，recommended 多任务配置 MUST 显式启用五模态和 auxiliary heads。

#### Scenario: 默认 fusion 配置保持 beam-only
- **WHEN** 用户加载现有 canonical fusion 配置
- **THEN** 配置 MUST 不默认启用遮挡或位置辅助任务
- **AND** 模型和 dataset MUST 保持现有 beam-only 行为

#### Scenario: 五模态多任务推荐配置
- **WHEN** 用户加载 recommended 五模态多任务 fusion 配置或 overlay
- **THEN** 配置 MUST 设置 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** 配置 MUST 启用 `cls_token_transformer_fusion` 的遮挡和位置辅助头
- **AND** 配置 MUST 启用 dataset 的遮挡和位置目标生成

#### Scenario: loss 权重可配置
- **WHEN** 用户在 fusion 配置中设置 beam、遮挡或位置 loss 权重
- **THEN** 训练流程 MUST 使用配置值计算多任务总 loss
- **AND** final config MUST 记录实际使用的权重

### Requirement: Fusion 配置校验多任务依赖
系统 MUST 对多任务 fusion 配置进行显式校验。启用遮挡目标时必须能访问 beam sweep power 文件；启用位置目标时必须声明位置目标来源；启用 auxiliary loss 时模型必须支持对应辅助输出。

#### Scenario: 位置目标缺少来源
- **WHEN** 配置启用位置辅助任务但未声明合法 `position_target_source`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 列出支持的 position target source

#### Scenario: 模型不支持辅助输出
- **WHEN** 配置启用遮挡或位置 loss，但 `model.student` 不支持对应 auxiliary head
- **THEN** 训练流程 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出模型输出缺少的辅助字段

#### Scenario: 遮挡目标与数据文件不匹配
- **WHEN** 配置启用遮挡目标但监督 beam 文件不是 64 维 power vector
- **THEN** dataset 构建或首次取样 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出遮挡标签生成依赖 64-beam sweep

### Requirement: Fusion objective 配置矩阵
系统 MUST 为 fusion 实验提供 objective-aware 配置入口，使同一模态集合能够分别运行 `beam`、`occlusion`、`position` 和 `multitask` 预测目标。配置命名 MUST 同时表达模态集合和预测目标。

#### Scenario: 五模态 objective 配置
- **WHEN** 用户查看 recommended 五模态 fusion 配置
- **THEN** 系统 MUST 提供或虚拟解析 beam、occlusion、position 和 multitask 四类 objective 入口
- **AND** 每个入口 MUST 使用相同的五模态集合 `[image, radar, gps, lidar, mmwave]`

#### Scenario: 配置名表达 objective
- **WHEN** 用户使用 objective-aware fusion 配置
- **THEN** 配置名或 virtual config stem MUST 包含 canonical 模态 slug 和 objective 名称
- **AND** 配置中的 `experiment.objective` MUST 与名称中的 objective 一致

#### Scenario: 旧配置兼容
- **WHEN** 用户继续使用既有 `configs/fusion/all_modalities_no_kd.yaml`
- **THEN** 系统 MUST 将该配置视为 `experiment.objective: beam`
- **AND** 系统 MUST 不要求用户修改旧运行命令

### Requirement: 模态失衡 objective 子集
fusion 配置系统 MUST 支持为模态失衡研究生成强模态、弱模态、单模态和全模态 objective 对照实验。每个 objective 配置 MUST 使用同一套 target 生成语义和同一套 metric 名称。

#### Scenario: strong-only occlusion 配置
- **WHEN** 用户请求 strong-only 模态集合的 occlusion fusion 配置
- **THEN** 系统 MUST 能生成只包含 strong modalities 的 fusion 配置
- **AND** 配置 MUST 设置 `experiment.objective: occlusion`

#### Scenario: weak-only position 配置
- **WHEN** 用户请求 weak-only 模态集合的 position fusion 配置
- **THEN** 系统 MUST 能生成只包含 weak modalities 的 fusion 配置
- **AND** 配置 MUST 设置 `experiment.objective: position`

#### Scenario: objective 间可比性
- **WHEN** 用户比较同一模态集合下的 beam、occlusion、position 和 multitask 结果
- **THEN** 系统 MUST 保持数据 split、target horizon、模态顺序和模型 backbone 默认配置一致

### Requirement: Objective-aware multitask canonical 默认等权
objective-aware fusion canonical 配置 MUST 在 `experiment.objective: multitask` 时默认使用 beam、occlusion 和 position 三个任务等权 loss。该默认值 MUST 应用于所有由 virtual canonical generator 生成的 multitask fusion 配置，包括 all-modalities、strong-only、weak-only 和显式模态 slug。

#### Scenario: 五模态 multitask 默认等权
- **WHEN** 开发者加载 `configs/fusion/image_radar_gps_lidar_mmwave_multitask_no_kd.yaml`
- **THEN** 解析后的配置 MUST 设置 `experiment.objective: multitask`
- **AND** 解析后的配置 MUST 启用 beam、occlusion 和 position 三类 targets 与 heads
- **AND** 解析后的配置 MUST 设置 `loss.objective.weights.beam: 1.0`
- **AND** 解析后的配置 MUST 设置 `loss.objective.weights.occlusion: 1.0`
- **AND** 解析后的配置 MUST 设置 `loss.objective.weights.position: 1.0`

#### Scenario: strong-only multitask 默认等权
- **WHEN** 开发者加载 `configs/fusion/strong_only_multitask_no_kd.yaml`
- **THEN** 解析后的配置 MUST 只包含 strong modalities `[gps, mmwave]`
- **AND** 解析后的配置 MUST 设置 beam、occlusion 和 position 三个 objective 权重均为 `1.0`

#### Scenario: weak-only multitask 默认等权
- **WHEN** 开发者加载 `configs/fusion/weak_only_multitask_no_kd.yaml`
- **THEN** 解析后的配置 MUST 只包含 weak modalities `[image, radar, lidar]`
- **AND** 解析后的配置 MUST 设置 beam、occlusion 和 position 三个 objective 权重均为 `1.0`

#### Scenario: 显式 multitask 权重覆盖
- **WHEN** 用户通过实体 YAML 或命令行覆盖显式设置 `loss.objective.weights.position`
- **THEN** 系统 MUST 使用用户显式配置的 position 权重
- **AND** 该覆盖 MUST 不改变未被覆盖的 beam 和 occlusion 权重

#### Scenario: multitask 权重记录到产物
- **WHEN** 完成 objective-aware multitask 训练
- **THEN** `final_config.yaml` 或等价 runtime metadata MUST 能追溯 beam、occlusion 和 position 的实际 loss 权重
- **AND** epoch log MUST 记录或能派生本次 multitask 总 loss 的权重组成

### Requirement: Snapshot fusion 配置
Fusion 配置体系 MUST 支持 snapshot next-frame no-KD baseline。该 baseline MUST 使用现有 `experiment.task: fusion` 输入路由、现有 `modalities` 标准化和现有 fusion batch 准备，但模型必须为无时序 snapshot fusion 模型。

#### Scenario: 五模态 snapshot fusion
- **WHEN** 用户加载五模态 snapshot fusion 配置
- **THEN** 配置 MUST 启用 `image`、`radar`、`gps`、`lidar` 和 `mmwave`
- **AND** dataset MUST 按现有 fusion 模态选择逻辑只读取启用模态
- **AND** 模型 MUST 对当前帧的五个模态表示执行融合
- **AND** 模型 MUST 输出 `[B, 1, num_classes]` logits

#### Scenario: 任意合法多模态 snapshot fusion
- **WHEN** 用户加载 `configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml` 且 `<slug>` 是两个到五个合法模态组成的 canonical slug
- **THEN** 系统 MUST 使用 `<slug>` 表示的模态集合构建 snapshot fusion
- **AND** forward MUST 只要求该模态集合对应的输入张量
- **AND** 未启用模态缺失不得阻止该配置运行

### Requirement: Snapshot fusion 不依赖 legacy fusion GRU
Snapshot fusion baseline MUST 不使用 `fusion_teacher`、`fusion_student` 的 GRU 路线作为主模型。若配置中保留 teacher 字段用于兼容结构，训练主模型 MUST 仍是无时序 snapshot 模型。

#### Scenario: no-KD snapshot fusion 主模型
- **WHEN** 用户训练 snapshot fusion no-KD 配置
- **THEN** 可训练主模型 MUST 为无时序 snapshot 模型
- **AND** 训练流程 MUST 不构建 frozen teacher checkpoint
- **AND** `distillation.teacher_model_name` MUST 为 `null`

#### Scenario: legacy fusion GRU 不参与 snapshot forward
- **WHEN** snapshot fusion 模型执行 forward
- **THEN** forward 路径 MUST 不调用 legacy `fusion_teacher` 或 `fusion_student` 的 GRU 层
- **AND** output diagnostics 或 final config MUST 标记 `uses_temporal_core: false`

### Requirement: Fusion 输入准备支持 CSI
训练、验证和评估流程在 `experiment.task: fusion` 下 MUST 能根据配置的 `modalities` 准备 CSI 输入。未启用 CSI 时，batch 准备和模型 forward MUST 不要求 `csi` 或 `csi_batch`。

#### Scenario: fusion 启用 CSI 和 GPS
- **WHEN** fusion 配置的 `modalities` 为 `["gps", "csi"]`
- **THEN** batch 准备 MUST 构造 `gps_batch` 和 `csi_batch`
- **AND** batch 准备 MUST 不要求 image、radar、LiDAR 或 mmWave 字段

#### Scenario: fusion 启用全部六模态
- **WHEN** fusion 配置的 `modalities` 为 `["image", "radar", "gps", "lidar", "mmwave", "csi"]`
- **THEN** batch 准备 MUST 构造六个模态输入
- **AND** 六个输入的 batch 和 sequence 维度 MUST 对齐

### Requirement: Modular fusion 使用 CSI encoder 输出
`modular_sequence` fusion 模型 MUST 能将 CSI encoder 的 `[B, T, D]` 输出与其它模态 encoder 输出对齐，并通过既有 projector 和 representation core 处理。

#### Scenario: modular_sequence 融合 CSI 与 mmWave
- **WHEN** 配置 `modalities: ["mmwave", "csi"]`
- **THEN** 模型 MUST 分别调用 mmWave encoder 和 CSI encoder
- **AND** 两个 projected feature MUST 在 batch、time 和 `d_model` 维度上兼容

### Requirement: Canonical fusion virtual config 不扩展 legacy KD 模式
Canonical fusion virtual config 生成器 MUST 聚焦当前 no-KD strong/lightweight/fusion 主线和仍被 active specs 批准的 objective/snapshot 入口。生成器 MUST 不再把 `logits_kd` 或 `rkd` 作为所有 fusion modality slug 的 canonical virtual mode；不存在实体 YAML 的 legacy KD fusion 路径 MUST 清晰失败。

#### Scenario: no-KD fusion virtual config 继续可用
- **WHEN** 用户请求当前支持的 no-KD canonical fusion virtual config
- **THEN** 配置加载器 MUST 生成完整配置
- **AND** 生成配置 MUST 包含当前主线模型、数据、训练和 lineage metadata
- **AND** 生成配置 MUST 不包含 KD-only temperature、alpha 或 RKD 权重字段

#### Scenario: KD fusion virtual config 不再接管路径
- **WHEN** 用户请求不存在实体 YAML 的 fusion `logits_kd` 或 `rkd` 配置
- **THEN** 配置加载器 MUST 拒绝该路径
- **AND** 错误信息 MUST 指向 legacy KD virtual alias 已退役，而不是生成或替换为其它配置

### Requirement: Legacy KD baseline 不影响 canonical 模态 slug 解析
删除 fusion KD virtual modes 后，canonical 模态 slug 解析 MUST 继续支持当前合法模态集合、顺序规范化、重复模态拒绝、未知模态拒绝和单模态转发建议。

#### Scenario: canonical slug 校验保持稳定
- **WHEN** 用户请求 no-KD fusion virtual config，并使用合法模态集合
- **THEN** 系统 MUST 按固定模态顺序解析 slug 并生成配置
- **AND** 重复模态、未知模态或可转为单模态配置的路径 MUST 继续给出清晰错误或建议

