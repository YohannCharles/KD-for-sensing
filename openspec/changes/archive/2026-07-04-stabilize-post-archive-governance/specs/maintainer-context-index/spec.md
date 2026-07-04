## ADDED Requirements

### Requirement: 维护索引 validation 命令必须引用当前可运行目标
`docs/maintainer_context_index.yaml` 中记录的 focused validation 命令 MUST 引用当前可运行的 OpenSpec spec、active change 或通用 strict 校验。若命令使用 `openspec validate <name> --strict` 形式，`<name>` MUST 是当前存在的 spec 名称、active change 名称，或被明确标记为历史记录且不得出现在可复制执行的 focused validation 列表中。归档后无 active change 时，索引 MUST 优先记录 `openspec validate --all --strict` 或 current spec validation，而不是已归档 change 的 validation 命令。

#### Scenario: focused validation 不引用已归档 change
- **WHEN** `openspec list --json` 不包含某个 change，且 `docs/maintainer_context_index.yaml` 的 focused validation 列表包含 `openspec validate <change> --strict`
- **THEN** 架构边界或文档健康检查 MUST 失败
- **AND** 失败信息 MUST 要求改用 `openspec validate --all --strict`、current spec validation 或恢复/说明 active change 状态

#### Scenario: 无 active change 时使用全量 OpenSpec strict
- **WHEN** `openspec list --json` 返回空 active change 列表
- **THEN** 维护索引的 focused validation MUST 包含 `openspec validate --all --strict` 或等价 current specs strict 校验
- **AND** 维护索引 MUST 不把 archive change validation 当作当前可执行验收入口

### Requirement: 维护索引统计基线必须声明口径
维护索引或其指向的 project surface inventory MUST 为架构尺寸基线声明统计来源、统计范围、是否包含未跟踪工作树文件、排除项和用途。统计数字 MUST 被描述为趋势定位和右尺寸化上下文，不得被解释为所有大文件必须拆分的硬 KPI。统计基线 MUST 明确排除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和其它本地产物。

#### Scenario: 统计口径可审计
- **WHEN** 维护者阅读维护索引或 inventory 的架构尺寸基线
- **THEN** 文档 MUST 能看出该数字来自当前工作树 on-disk 扫描、tracked-only 扫描、CodeGraph/AST 历史基线或其它明确来源
- **AND** 文档 MUST 说明扫描范围和排除的本地产物路径

#### Scenario: 数量漂移需要 rationale
- **WHEN** Python 文件数、测试文件数、脚本数量或 YAML 数量与上一轮基线明显不同
- **THEN** 维护索引或 inventory MUST 记录变化来自新增 current capability、热点拆分、helper 合并、生成型配置删除、未跟踪 helper 或治理漂移中的哪类原因
- **AND** 健康检查 MUST 不仅根据数量变化要求机械拆分或回退源码
