## ADDED Requirements

### Requirement: 训练编排层保持窄职责
训练主循环 MUST 只协调 epoch、checkpoint、optimizer、scheduler、extension hook、validation 调用和运行产物写出。objective metric alias、available metric 计算、TensorBoard objective 字段、validation forward/loss/collect 和 canonical overlay 生成 MUST 位于对应窄模块。

#### Scenario: 新增 objective 不修改 trainer 主循环
- **WHEN** 开发者新增一个 prediction objective 并完成 objective metadata、loss 和 metrics 实现
- **THEN** 不得要求修改 trainer 主循环中的 early stopping alias 表、history 字段表或 TensorBoard objective 字段表
- **AND** trainer MUST 通过 objective metadata 自动记录该 objective 的 primary metric 和日志字段

#### Scenario: 修改 validation 指标不修改 trainer 主循环
- **WHEN** 开发者修复 validation pass 中某个 objective 指标的聚合方式
- **THEN** 变更 MUST 限定在 evaluation pass、objective metrics 或 evaluation metrics 模块
- **AND** 不需要编辑 trainer 主循环

### Requirement: 启用模态解析唯一来源
训练、验证、评估、诊断和 dataset 构建路径 MUST 使用 `engine.modality_resolution` 或其公开 helper 解析启用模态。入口层不得新增 `_uses_gps`、`_uses_lidar`、`_uses_mmwave` 等重复配置推导 helper。

#### Scenario: evaluator 复用 modality resolution
- **WHEN** 评估入口需要判断当前配置是否启用 LiDAR 或 mmWave
- **THEN** 入口 MUST 调用统一模态解析 helper
- **AND** 不得在 evaluator 中维护独立的配置字段判断逻辑

#### Scenario: fusion teacher/student 模态冲突错误一致
- **WHEN** fusion 配置中 teacher 和 student modalities 不一致且未声明支持跨模态蒸馏
- **THEN** 训练和评估路径 MUST 抛出一致的错误信息
- **AND** 错误 MUST 来自统一模态解析逻辑

### Requirement: models 包级延迟导出
`kd_sensing.models` MUST 保持公开符号兼容，同时通过延迟导入暴露重依赖模型类。导入 `kd_sensing.models` 本身 MUST 不 eager import fusion、GPS、LiDAR、mmWave、image encoder 或其它模型实现模块。

#### Scenario: 轻量导入 models 包
- **WHEN** 开发者执行 `import kd_sensing.models`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入各模型实现模块

#### Scenario: 按需访问公开模型符号
- **WHEN** 开发者执行 `from kd_sensing.models import FusionTeacherModalityNet`
- **THEN** 系统 MUST 按需导入对应实现模块并返回该公开符号
- **AND** `__all__` 中的既有公开模型符号 MUST 继续可访问

#### Scenario: removed alias 错误保持兼容
- **WHEN** 现有代码访问已移除的模型别名
- **THEN** `kd_sensing.models` MUST 继续抛出清晰 `AttributeError`
- **AND** 错误信息 MUST 指向替代公开符号
