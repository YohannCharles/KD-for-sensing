## ADDED Requirements

### Requirement: Fusion canonical 多模态配置矩阵
项目 MUST 为 `image`、`radar`、`gps`、`lidar` 的所有必要多模态组合提供 canonical fusion 配置矩阵。多模态组合 MUST 覆盖全部 6 个双模态组合、4 个三模态组合和 1 个四模态组合。每个组合 MUST 提供 teacher no-KD、student no-KD、logits KD 和 RKD 配置。

#### Scenario: 双模态 fusion 组合完整
- **WHEN** 开发者查看 `configs/fusion/`
- **THEN** 系统 MUST 提供 `image_radar`、`image_gps`、`image_lidar`、`radar_gps`、`radar_lidar` 和 `gps_lidar` 六个双模态 slug 的 canonical 配置
- **AND** 每个 slug MUST 具备 `<slug>_teacher_no_kd.yaml`、`<slug>_student_no_kd.yaml`、`<slug>_logits_kd.yaml` 和 `<slug>_rkd.yaml`

#### Scenario: 三模态 fusion 组合完整
- **WHEN** 开发者查看 `configs/fusion/`
- **THEN** 系统 MUST 提供 `image_radar_gps`、`image_radar_lidar`、`image_gps_lidar` 和 `radar_gps_lidar` 四个三模态 slug 的 canonical 配置
- **AND** 每个 slug MUST 具备 `<slug>_teacher_no_kd.yaml`、`<slug>_student_no_kd.yaml`、`<slug>_logits_kd.yaml` 和 `<slug>_rkd.yaml`

#### Scenario: 四模态 fusion 组合完整
- **WHEN** 开发者查看 `configs/fusion/`
- **THEN** 系统 MUST 提供 `image_radar_gps_lidar_teacher_no_kd.yaml`、`image_radar_gps_lidar_student_no_kd.yaml`、`image_radar_gps_lidar_logits_kd.yaml` 和 `image_radar_gps_lidar_rkd.yaml`

#### Scenario: 不重复提供 fusion 单模态入口
- **WHEN** 用户需要运行单模态 image、radar、GPS 或 LiDAR 实验
- **THEN** 文档 MUST 引导用户使用 `configs/<modality>/` 下的单模态 canonical 配置
- **AND** fusion canonical 矩阵 MUST 不要求提供单模态 fusion duplicate 配置

### Requirement: Fusion canonical 配置语义
每个 canonical fusion 配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar` 生成 slug，并 MUST 让 teacher 和 student 使用相同的 `modalities`。同一 slug 的四种配置 MUST 只改变训练角色和 KD 模式，不得改变模态集合。

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
canonical fusion 配置 MUST 根据 `modalities` 启用对应 dataset 字段，并不得要求未启用模态的数据列。启用 GPS 的配置 MUST 使用 GPS-Rel-Polar；启用 LiDAR 的配置 MUST 使用 LiDAR BEV 默认字段。

#### Scenario: 启用 GPS 的 fusion 配置
- **WHEN** canonical fusion 配置的 `modalities` 包含 `gps`
- **THEN** 配置 MUST 设置 `data.dataset.use_gps: true`
- **AND** 配置 MUST 设置 `gps_feature_mode: relative_polar`
- **AND** teacher 和 student 的 `gps_input_size` MUST 为 3

#### Scenario: 启用 LiDAR 的 fusion 配置
- **WHEN** canonical fusion 配置的 `modalities` 包含 `lidar`
- **THEN** 配置 MUST 设置 `data.dataset.use_lidar: true`
- **AND** 配置 MUST 提供 LiDAR BEV size、ROI 和 normalize 默认字段
- **AND** teacher 和 student MUST 使用与 LiDAR BEV 输入通道一致的 `lidar_channels`

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
