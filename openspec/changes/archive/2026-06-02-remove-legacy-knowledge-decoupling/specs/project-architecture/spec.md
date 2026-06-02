## MODIFIED Requirements

### Requirement: 源码与实验产物边界
项目 MUST 明确源码、配置、文档、OpenSpec artifacts 与本地数据、训练日志、缓存和输出产物的边界。本地运行产物 MUST 保持在 `.gitignore` 覆盖范围内，文档 MUST 指明哪些目录是可复现输入、哪些目录是可删除生成物。用户明确要求退役并清理某条失败实验路线时，系统 MAY 删除匹配的本地 `outputs/`、`logs/`、cache、checkpoint 和训练诊断产物，但 MUST 先生成可审计清单并限制在未纳入源码的运行产物内。

#### Scenario: 本地产物不进入版本控制
- **WHEN** 用户运行训练、评估、预处理或诊断命令
- **THEN** 生成的 logs、outputs、cache、checkpoint 和 Python bytecode 产物 MUST 位于忽略规则覆盖的路径或文件模式内
- **AND** 项目文档 MUST 不要求提交这些本地产物

#### Scenario: 文档说明产物边界
- **WHEN** 开发者阅读 README 或扩展指南
- **THEN** 文档 MUST 说明 `dataset/`、`All_models/`、`outputs/`、`logs/` 和 cache 目录的角色
- **AND** 文档 MUST 指明哪些目录通常不应纳入源码变更
- **AND** 文档 MUST 明确用户未要求清理时，源码删除不应自动清理历史 `outputs/`

#### Scenario: 清理旧失败实验产物
- **WHEN** 用户明确要求删除已退役失败路线的输出日志和实验结果
- **THEN** 清理流程 MUST 先写出 machine-readable manifest，记录每个候选路径、匹配原因、产物类型和大小
- **AND** 清理流程 MUST NOT 删除 `dataset/`、`All_models/` 已跟踪权重、OpenSpec artifacts、源码文件或未匹配失败路线的活跃实验产物
