## MODIFIED Requirements

### Requirement: 虚拟 canonical fusion 配置解析
系统 MUST 支持从 canonical fusion 配置路径生成配置，即使该路径在磁盘上没有实体 YAML 文件。可生成路径 MUST 仅限当前 supervised/adaptation 入口，例如 `configs/fusion/<slug>_strong.yaml`、`configs/fusion/<slug>_lightweight.yaml`、snapshot 或 active overlay recipe。系统 MUST 不再生成 `logits_kd`、`rkd` 或包含 `distillation` block 的配置。

#### Scenario: 加载 supervised canonical fusion 配置路径
- **WHEN** 用户加载缺失但合法的 `configs/fusion/gps_mmwave_lightweight.yaml`
- **THEN** 系统 MUST 解析该路径并生成可用于训练、评估和测试的最终配置
- **AND** 最终配置的 `experiment.task` MUST 为 `fusion`
- **AND** 最终配置 MUST 不包含 `distillation` 配置块

#### Scenario: KD virtual path 被拒绝
- **WHEN** 用户加载 `configs/fusion/gps_mmwave_logits_kd.yaml` 或 `configs/fusion/gps_mmwave_rkd.yaml`
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 系统 MUST 不生成配置或回退为 lightweight 配置

### Requirement: 生成配置语义
虚拟 canonical fusion 配置 MUST 生成当前 supervised/adaptation 语义，包括任务类型、模态启用字段、primary 模型配置、loss、训练参数和输出 run name。生成配置 MUST 不包含 teacher checkpoint 来源、distillation type、temperature、alpha 或 RKD 参数。

#### Scenario: 生成 strong fusion 配置
- **WHEN** 用户加载 `configs/fusion/gps_lidar_mmwave_strong.yaml`
- **THEN** 最终配置 MUST 设置 `experiment.task: fusion`
- **AND** `model.primary` MUST 表示 strong fusion baseline
- **AND** 配置 MUST 启用 GPS、LiDAR 和 mmWave 对应的数据及模型输入字段

#### Scenario: 生成 lightweight fusion 配置
- **WHEN** 用户加载 `configs/fusion/gps_lidar_mmwave_lightweight.yaml`
- **THEN** 最终配置 MUST 设置 `experiment.task: fusion`
- **AND** `model.primary` MUST 表示 lightweight 或 CLS-token fusion 主模型
- **AND** 最终配置 MUST 不包含 `distillation.teacher_model_name`

