## MODIFIED Requirements

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
