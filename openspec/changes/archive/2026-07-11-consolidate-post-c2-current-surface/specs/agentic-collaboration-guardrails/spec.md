## MODIFIED Requirements

### Requirement: 本地/手动验证和调度友好检查
项目 MUST 提供本地/手动验证策略。默认 quick verify MUST 保持无真实数据、无训练、无 checkpoint；OpenSpec、architecture、CLI/config、compile 和小型安全扫描 MAY 由人工或任务系统组合运行。项目 MUST 不要求 surface doctor 或 research preview 作为验证编排层。

#### Scenario: Quick verify 无训练
- **WHEN** 开发者或 agent 运行 quick verify
- **THEN** 检查 MUST 运行 OpenSpec strict 和架构边界或等价 quick checks
- **AND** MUST 不启动真实训练、不读取 dataset、不加载 checkpoint

#### Scenario: Focused checks 直接报告漂移
- **WHEN** 人工运行 security、CLI/config 或 compile check
- **THEN** 原生命令 MUST 报告路径和失败原因
- **AND** 输出 MUST 不修改源码、OpenSpec、本地产物或用户改动

### Requirement: 脏工作树和 OpenSpec 收口 preflight
协作流程 MUST 通过 `git status --short`、`openspec list --json` 和必要 status/validate 命令报告 active change、complete 未归档 change、dirty tracked files 和 runtime artifact 边界。Preflight MUST 不依赖 project surface doctor/research preview，不自动 archive、不删除、不 reset、不覆盖用户改动。

#### Scenario: Active/complete change 状态清晰
- **WHEN** preflight 发现 complete 未归档或重叠 active change
- **THEN** implementation MUST 记录 archive、abandon、scope correction 或 deferral 决策
- **AND** MUST 不把其过时 artifact 当作 current implementation 事实

#### Scenario: Dirty worktree 不被自动清理
- **WHEN** git status 显示用户改动
- **THEN** implementation MUST 记录并避开该 diff
- **AND** MUST NOT reset、checkout、删除或覆盖文件
