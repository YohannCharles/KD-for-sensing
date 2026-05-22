## ADDED Requirements

### Requirement: Objective 元数据轻量导入边界
项目 MUST 将 prediction objective 的纯元数据契约与 torch loss/runtime 实现解耦。导入配置加载、配置 normalization 或配置 validation 路径时，系统 MUST 能读取 objective 默认 metric、metric mode、required target/head、history fields 和 TensorBoard scalar 映射，且不得因此导入 torch、模型、dataset、诊断渲染或训练主循环。

#### Scenario: 配置导入不触发 torch
- **WHEN** 开发者执行 `import kd_sensing.config`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `torch`
- **AND** 系统 MUST 不导入 dataset 类、模型实现、诊断可视化 core 或训练主循环

#### Scenario: runtime loss 仍可通过原入口使用
- **WHEN** 训练或验证代码从 `kd_sensing.engine.prediction_objectives` 导入 prediction target 和 loss helper
- **THEN** 导入 MUST 继续成功
- **AND** torch loss 计算语义 MUST 与变更前保持兼容
- **AND** objective 元数据 MUST 来自同一轻量契约，避免配置路径和 runtime 路径维护两套表

### Requirement: 诊断可视化轻量 helper import 边界
诊断可视化内部模块 MUST 按职责控制 import 边界。配置解析、采样选择、metadata 写出和 JSON 序列化 helper MUST 不导入 matplotlib、PIL、dataset builder、model builder 或训练 runtime。渲染、数据集构建和模型预测导出等重依赖职责 MUST 留在对应重模块或函数内部。

#### Scenario: 导入可视化配置 helper 不触发渲染栈
- **WHEN** 开发者执行 `import kd_sensing.diagnostics.visualization.config`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `matplotlib`
- **AND** 系统 MUST 不导入 `PIL.Image`
- **AND** 系统 MUST 不导入 `kd_sensing.engine.data_factory`

#### Scenario: 导入采样和写出 helper 不构建数据集
- **WHEN** 开发者导入 `kd_sensing.diagnostics.visualization.sampling` 或 `kd_sensing.diagnostics.visualization.writers`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 dataset builder、model builder 或 visualization core
- **AND** 这些模块 MUST 只在调用实际需要的数据结构或文件写出逻辑时处理输入参数

### Requirement: OpenSpec 文档健康检查结构化
项目 MUST 使用结构化方式检查 OpenSpec capability purpose。健康检查 MUST 只检查每个 spec 的 `## Purpose` 段落是否为空、过短或仍为归档占位文本，不得因为正文中描述被拒绝的占位文本而误判。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: purpose 检查不自引用误伤
- **WHEN** 某个 spec 正文描述健康检查会拒绝归档占位文本
- **THEN** 健康检查 MUST 不因正文出现该字符串而失败
- **AND** 检查 MUST 只根据 `## Purpose` 段落判断该 spec 是否存在文档健康问题

#### Scenario: purpose 问题报告具体 spec
- **WHEN** 某个 spec 的 `## Purpose` 段落为空、过短或仍为归档占位文本
- **THEN** 健康检查 MUST 报告该 spec 路径
- **AND** 报告 MUST 指向需要补齐的 capability purpose，而不是要求改写无关正文
