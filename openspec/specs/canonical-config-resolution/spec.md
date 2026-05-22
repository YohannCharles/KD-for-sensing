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

### Requirement: canonical overlay recipe 化
canonical fusion 配置和 advanced overlay 生成 MUST 由可审查的 recipe/table 驱动。objective、G2D、CRAF、MARF 和通用 fusion overlay MUST 按职责拆分定义，`build_virtual_config()` 入口 MUST 只负责路径识别、recipe 查找和应用。

#### Scenario: 既有 canonical 路径生成语义不变
- **WHEN** 用户加载既有 virtual canonical fusion 路径
- **THEN** 系统 MUST 通过 recipe 生成与变更前等价的关键配置语义
- **AND** experiment name、task、modalities、student/teacher model、distillation、loss、training 和 output run name MUST 保持兼容

#### Scenario: objective overlay recipe
- **WHEN** 用户加载 `configs/fusion/gps_mmwave_occlusion_no_kd.yaml`
- **THEN** 系统 MUST 通过 objective recipe 生成 `experiment.objective: occlusion`
- **AND** recipe MUST 启用 occlusion target、occlusion head、objective loss 和对应 early stopping 默认值

#### Scenario: advanced overlay recipe 错误可诊断
- **WHEN** 用户加载未知 advanced overlay 路径
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 列出可用 overlay recipe 名称

### Requirement: config/io 职责收敛
配置加载流程 MUST 将 config source 解析、命令行 overlay、normalization pipeline、migration guard、dataset-specific rules 和 schema validation 拆到明确 helper。`config/io.py` MAY 作为入口协调这些 helper，但 MUST 不手写 canonical overlay、dataset 专属规则、removed feature guard 或 objective 专属默认表的主要实现。

#### Scenario: 加载实体配置
- **WHEN** 用户加载磁盘上存在的 YAML 配置
- **THEN** `config/io.py` MUST 读取实体文件并应用覆盖、默认补全和校验
- **AND** MUST 不进入 canonical recipe 生成路径
- **AND** 后续 normalization 和 validation MUST 通过明确 helper 执行

#### Scenario: 加载 virtual 配置
- **WHEN** 用户加载缺失但合法的 canonical virtual 配置路径
- **THEN** `config/io.py` MUST 调用 canonical recipe 入口获得基础配置
- **AND** 后续覆盖、默认补全和校验流程 MUST 与实体配置一致

#### Scenario: dataset 专属规则位于 dataset rule helper
- **WHEN** 用户加载 Raymobtime s008、DeepSense6G snapshot 或其它 dataset 专属配置
- **THEN** dataset 专属配置约束 MUST 由 dataset rule helper 或等价 validation helper 执行
- **AND** `config/io.py` MUST 不直接维护该 dataset 的完整业务规则

#### Scenario: migration guard 独立于 io 入口
- **WHEN** 用户配置已删除的 image motion profile、image cache、legacy encoder 或其它迁移拒绝项
- **THEN** 系统 MUST 由 migration guard 或等价 helper 抛出清晰错误
- **AND** `config/io.py` MUST 不直接维护所有已删除选项的完整拒绝逻辑

#### Scenario: normalization 顺序可测试
- **WHEN** 开发者对实体配置、virtual canonical 配置、snapshot 配置或 Raymobtime 配置运行 config load 等价测试
- **THEN** 测试 MUST 能验证 source、overlay、normalization 和 validation 的执行顺序
- **AND** 命令行覆盖 MUST 继续在配置生成后生效，并在必要的 runtime requirement 校验前被考虑

### Requirement: Snapshot canonical 配置解析
配置加载流程 MUST 能识别并生成 snapshot next-frame baseline 配置。可生成路径 MUST 包含单模态 `configs/<modality>/snapshot_next_frame_no_kd.yaml` 和 fusion `configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml`。

#### Scenario: 生成单模态 snapshot 配置
- **WHEN** 用户加载缺失但合法的 `configs/gps/snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 生成可用于训练和评估的 GPS snapshot 配置
- **AND** 最终配置 MUST 设置 `experiment.task: gps`
- **AND** 最终配置 MUST 设置 `data.dataset.seq_len: 1` 和 `data.dataset.num_pred: 1`
- **AND** 最终配置 MUST 设置 `data.dataset.train_csv_name: train_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** 最终配置 MUST 设置 `data.dataset.val_csv_name: val_seqs_SNAPSHOT_NEXT_FRAME.csv` 或等价 validation CSV 字段
- **AND** 最终配置 MUST 构建 `snapshot_frame` core

#### Scenario: 生成 fusion snapshot 配置
- **WHEN** 用户加载缺失但合法的 `configs/fusion/gps_mmwave_snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 生成可用于训练和评估的 fusion snapshot 配置
- **AND** 最终配置 MUST 设置启用模态为 `["gps", "mmwave"]`
- **AND** 最终配置 MUST 设置 `experiment.task: fusion`
- **AND** 最终配置 MUST 构建无时序 snapshot fusion 模型

#### Scenario: 拒绝非法 snapshot slug
- **WHEN** 用户加载 `configs/fusion/mmwave_gps_snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 提示 canonical slug 为 `gps_mmwave`

### Requirement: Snapshot 配置生成语义
生成的 snapshot 配置 MUST 明确覆盖历史窗口默认值，并保持实体 YAML 优先、命令行覆盖后应用和 schema 校验流程一致。

#### Scenario: 覆盖历史窗口默认值
- **WHEN** 系统生成任一 snapshot 配置
- **THEN** 生成配置 MUST 覆盖默认 `seq_len=8` 和 `num_pred=3`
- **AND** 生成配置 MUST 覆盖默认 GRU representation core
- **AND** 生成配置 MUST 覆盖默认历史窗口 CSV 为 snapshot 专用 train/validation CSV
- **AND** 生成配置 MUST 设置 `output.run_name` 包含 `snapshot_next_frame_no_kd`

#### Scenario: 实体 snapshot 配置优先
- **WHEN** 用户加载磁盘上存在的 snapshot YAML
- **THEN** 配置加载流程 MUST 使用实体 YAML 内容
- **AND** virtual snapshot 规则 MUST 不覆盖同名实体配置

#### Scenario: 命令行覆盖仍生效
- **WHEN** 用户加载 snapshot virtual 配置并传入覆盖项 `training.epochs=1`
- **THEN** 最终配置 MUST 使用 `training.epochs: 1`
- **AND** snapshot 必需字段 MUST 继续满足 `seq_len=1`、`num_pred=1` 和无时序 core 契约，除非用户显式退出 snapshot 变体并通过校验

