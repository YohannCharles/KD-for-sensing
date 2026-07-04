## ADDED Requirements

### Requirement: 归档后版本控制状态必须成对审计
OpenSpec change 归档后，版本控制状态 MUST 能审计 active change 目录删除与 archive 目录新增是否成对提交。若 `git status --short` 同时显示 `openspec/changes/<change>/` 下文件被删除，并显示 `openspec/changes/archive/<date>-<change>/` 为未跟踪或新增，则实现任务、架构边界测试或最终说明 MUST 要求二者作为同一收口状态处理，或记录明确 deferral 原因。

#### Scenario: 删除 active change 但 archive 未纳入提交
- **WHEN** git status 显示 `D openspec/changes/<change>/...`
- **AND** 同名 dated archive 目录存在于 `openspec/changes/archive/<date>-<change>/`
- **THEN** 归档收口检查 MUST 报告该 change 处于待成对提交状态
- **AND** 维护者 MUST 将 active 目录删除和 archive 新增一起纳入提交，或在最终说明中记录暂缓原因

#### Scenario: archive 目录不代表 active requirement
- **WHEN** `openspec list --json` 不包含某个 change，但 `openspec/changes/archive/<date>-<change>/` 存在或未跟踪
- **THEN** agent 和维护者 MUST 将其解释为归档后版本控制收口状态
- **AND** 不得把该 archive 目录当作仍在实施的 active change 或 current requirement 来源

#### Scenario: 归档成对检查无运行时副作用
- **WHEN** 开发者运行归档完整性检查或架构边界测试
- **THEN** 检查 MUST 只读取 OpenSpec metadata、git-tracked path 状态和文档
- **AND** 检查 MUST 不删除、移动、压缩或重写 active change、archive 目录、`dataset/`、`outputs/`、`logs/`、cache 或 checkpoint
