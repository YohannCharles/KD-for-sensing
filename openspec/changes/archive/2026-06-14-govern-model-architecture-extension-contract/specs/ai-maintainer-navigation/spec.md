## ADDED Requirements

### Requirement: 模型架构扩展导航
AI 维护导航文档 MUST 在模型、forward、registry 或 baseline 改动路由中指向 `model-architecture-extension-contract`、`modular-sequence-model`、`component-registry`、共享 batch/runtime 和相关 focused tests。AI agent MUST 在非平凡模型改动前判断改动属于 config-only、component baseline、whole-model exception 还是 workflow/paper reproduction。

#### Scenario: AI 新增普通 baseline 前检查契约
- **WHEN** AI agent 准备新增或修改普通 supervised/adaptation baseline
- **THEN** 导航文档 MUST 要求优先选择 `modular_sequence` 配置或子组件注册路径
- **AND** agent MUST 不直接新增整模型注册名，除非 change artifact 明确 whole-model exception

#### Scenario: AI 新增论文复现 workflow 前检查边界
- **WHEN** AI agent 准备新增 paper reproduction 或多阶段 baseline workflow
- **THEN** 导航文档 MUST 指向 `src/kd_sensing/baselines/<family>/`、包内 CLI、脚本 allowlist 和本地产物边界
- **AND** agent MUST 不复制通用训练循环或新增 root-level 旧式入口

### Requirement: 模型改动验证路由更具体
模型相关任务路由 MUST 区分模块化组件、整模型例外、batch metadata 和 workflow baseline 的验证命令。至少 MUST 提到架构边界测试、对应模型 focused forward tests、配置加载 characterization，以及触碰 reliability metadata 时的 difficulty/batch tests。

#### Scenario: reliability-aware 模型改动验证
- **WHEN** 模型改动声明消费 observability/reliability metadata
- **THEN** 导航文档 MUST 建议运行相关 batch/difficulty focused tests
- **AND** 验证说明 MUST 覆盖普通 baseline 忽略 metadata 与 opt-in 模型接收 metadata 两种路径
