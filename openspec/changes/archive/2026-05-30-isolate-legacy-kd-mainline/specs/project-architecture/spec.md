## ADDED Requirements

### Requirement: Active mainline 与 legacy KD 模块边界
项目 MUST 区分 active mainline 方法模块和 legacy KD/baseline 模块。active mainline 包括当前推荐的 supervised beam prediction、HiST-Beam 跨场景适配、history-anchored residual、adapter/prototype/calibration、soft-label supervised training 和 LOSO summary；这些模块 MUST 不依赖 legacy KD runtime 聚合入口。

#### Scenario: mainline 导入不触发 KD runtime
- **WHEN** 开发者导入 active mainline 的训练、评估、HiST-Beam LOSO、history residual 或 soft-label helper
- **THEN** 导入 MUST 不构建 frozen teacher runtime
- **AND** 导入 MUST 不解析 teacher checkpoint registry
- **AND** 导入 MUST 不要求 legacy KD baseline 模块可用

#### Scenario: 架构测试拒绝 KD 回流
- **WHEN** 内部源码新增 active mainline 到 legacy KD runtime 聚合入口的依赖
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向 no-KD objective、method extension 或 explicit legacy baseline adapter 作为修复路径

### Requirement: Distillation 保留层保持纯算法职责
若 `kd_sensing.distillation` 继续保留，项目 MUST 将其作为纯算法或 legacy baseline 支撑层，而不是训练对象构建层。该层 MUST 不负责读取 dataset、构建 model、解析 checkpoint、选择 device、创建 optimizer 或写出 run artifact。

#### Scenario: distillation 轻量导入
- **WHEN** 开发者导入 `kd_sensing.distillation` 中保留的 loss、schedule 或 tensor helper
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 dataset builder、model builder、checkpoint loader 或训练主循环

#### Scenario: teacher checkpoint 解析不在算法层
- **WHEN** legacy KD baseline 需要加载 teacher checkpoint
- **THEN** checkpoint 解析 MUST 位于 engine runtime、checkpoint utility 或 legacy baseline adapter
- **AND** distillation 算法模块 MUST 只接收已经准备好的 teacher/student 张量或特征

### Requirement: 新主线方法默认无需 distillation 配置段
新 active mainline 配置和运行时 MUST 能在没有 KD-specific 字段的情况下表达 supervised/adaptation 训练。为了兼容旧配置，系统 MAY 接受 `distillation.type: no_kd`，但不得要求 `temperature`、`alpha`、`rkd_*` 或 `teacher_model_name` 作为 no-KD 主线的必要字段。

#### Scenario: no-KD 配置无需 teacher 字段
- **WHEN** 用户加载当前推荐的 no-KD mainline 配置
- **THEN** 配置 validation MUST 不要求 `distillation.teacher_model_name`
- **AND** 配置 validation MUST 不要求 KD temperature、alpha 或 RKD 权重字段

#### Scenario: 旧 no_kd 字段兼容
- **WHEN** 用户加载仍包含 `distillation.type: no_kd` 的历史配置
- **THEN** 系统 MUST 继续按普通 supervised/adaptation 训练处理
- **AND** 系统 MUST 不把该 run 标记为 KD baseline
