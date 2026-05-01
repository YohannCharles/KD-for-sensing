# canonical-config-resolution Specification

## Purpose
TBD - created by archiving change virtual-canonical-configs. Update Purpose after archive.
## Requirements
### Requirement: 虚拟 canonical fusion 配置解析
系统 MUST 支持从 canonical fusion 配置路径生成配置，即使该路径在磁盘上没有实体 YAML 文件。可生成路径 MUST 仅限 `configs/fusion/<slug>_<mode>.yaml`，其中 `<mode>` MUST 是 `teacher_no_kd`、`student_no_kd`、`logits_kd` 或 `rkd`。

#### Scenario: 加载不存在的 canonical fusion 配置路径
- **WHEN** 用户加载 `configs/fusion/gps_mmwave_logits_kd.yaml` 且该文件不存在
- **THEN** 系统 MUST 解析该路径并生成可用于训练、评估和测试的最终配置
- **AND** 最终配置的 `experiment.name` 和 `output.run_name` MUST 为 `gps_mmwave_logits_kd`
- **AND** 最终配置的 `experiment.task` MUST 为 `fusion`

#### Scenario: 非 canonical 缺失文件不自动生成
- **WHEN** 用户加载 `configs/custom/missing.yaml` 或 `configs/fusion/not_a_canonical_name.yaml` 且该文件不存在
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 系统 MUST 不把任意缺失 YAML 当作可生成配置

#### Scenario: 实体配置文件优先
- **WHEN** 用户加载一个磁盘上存在的 YAML 配置文件
- **THEN** 系统 MUST 按实体 YAML 内容加载配置
- **AND** 系统 MUST 不用同名虚拟 canonical 规则覆盖该实体文件

### Requirement: canonical fusion slug 命名校验
系统 MUST 使用固定模态优先级 `image > radar > gps > lidar > mmwave` 解析 fusion slug。slug MUST 由两个到五个不同合法模态组成；乱序、重复、未知模态和单模态 fusion slug MUST 被拒绝。

#### Scenario: 按 canonical 顺序解析 slug
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_rkd.yaml`
- **THEN** 系统 MUST 将 slug 解析为 `["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** teacher 和 student 的 `modalities` MUST 使用相同顺序

#### Scenario: 拒绝乱序 slug
- **WHEN** 用户加载 `configs/fusion/mmwave_gps_logits_kd.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 提示 canonical slug 为 `gps_mmwave`

#### Scenario: 拒绝重复模态 slug
- **WHEN** 用户加载 `configs/fusion/image_image_rkd.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 指出 fusion slug 不能包含重复模态

#### Scenario: 拒绝未知模态 slug
- **WHEN** 用户加载 `configs/fusion/image_wifi_logits_kd.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 包含非法模态名称 `wifi`

#### Scenario: 拒绝单模态 virtual fusion slug
- **WHEN** 用户加载 `configs/fusion/mmwave_student_no_kd.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 引导用户使用 `configs/mmwave/student_no_kd.yaml`

### Requirement: 生成配置语义
虚拟 canonical fusion 配置 MUST 生成与旧实体 canonical YAML 等价的核心语义，包括任务类型、模态启用字段、teacher/student 模型配置、KD 模式、训练参数和默认 teacher checkpoint 来源。

#### Scenario: 生成 teacher no-KD fusion 配置
- **WHEN** 用户加载 `configs/fusion/gps_lidar_mmwave_teacher_no_kd.yaml`
- **THEN** 最终配置 MUST 设置 `distillation.type: no_kd`
- **AND** `model.student.type` MUST 为 `fusion_teacher`
- **AND** teacher 和 student 的 `modalities` MUST 为 `["gps", "lidar", "mmwave"]`
- **AND** 配置 MUST 启用 GPS、LiDAR 和 mmWave 对应的数据及模型输入字段

#### Scenario: 生成 student no-KD fusion 配置
- **WHEN** 用户加载 `configs/fusion/gps_lidar_mmwave_student_no_kd.yaml`
- **THEN** 最终配置 MUST 设置 `distillation.type: no_kd`
- **AND** `model.student.type` MUST 为 `fusion_student`
- **AND** `distillation.teacher_model_name` MUST 为 `null`

#### Scenario: 生成 logits KD fusion 配置
- **WHEN** 用户加载 `configs/fusion/gps_mmwave_logits_kd.yaml`
- **THEN** 最终配置 MUST 设置 `distillation.type: logits_kd`
- **AND** `model.teacher.type` MUST 为 `fusion_teacher`
- **AND** `model.student.type` MUST 为 `fusion_student`
- **AND** 默认 teacher checkpoint MUST 指向同 slug teacher no-KD 的 `best.pth`

#### Scenario: 生成 RKD fusion 配置
- **WHEN** 用户加载 `configs/fusion/radar_lidar_mmwave_rkd.yaml`
- **THEN** 最终配置 MUST 设置 `distillation.type: rkd`
- **AND** 最终配置 MUST 包含 RKD pair、distance weight 和 angle weight 参数
- **AND** teacher 和 student MUST 使用相同 `modalities`

#### Scenario: 保持 image+radar 兼容参数
- **WHEN** 用户加载 `configs/fusion/image_radar_logits_kd.yaml`
- **THEN** 最终配置 MUST 保持 image+radar upstream 兼容参数
- **AND** fusion teacher GRU MUST 为 `[64, 64, 2]`
- **AND** fusion student GRU MUST 为 `[64, 64, 1]`
- **AND** 默认 teacher checkpoint MUST 使用 `All_models/BothTeacher_best.pth`

#### Scenario: 命令行覆盖应用在生成配置之后
- **WHEN** 用户加载虚拟 canonical fusion 配置并传入覆盖项 `training.epochs=1`
- **THEN** 最终配置 MUST 使用 `training.epochs: 1`
- **AND** 其它由 canonical 规则生成的字段 MUST 保持有效

