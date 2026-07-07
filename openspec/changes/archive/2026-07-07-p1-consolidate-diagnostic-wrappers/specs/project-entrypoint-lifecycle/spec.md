## ADDED Requirements

### Requirement: 薄诊断 wrapper 必须收敛到领域 owner
项目 MUST 不长期保留只调用同一领域 owner 的 plot、compare、visualize、recommend 或 prepare wrapper。若 wrapper 没有独立输入契约、输出 schema 或 claim gate，它 MUST 合并为领域 owner CLI 的 subcommand、mode flag 或 documented command recipe。

#### Scenario: wrapper 只有转发职责
- **WHEN** 一个 CLI 或 script 只解析少量参数并调用同一 owner function
- **THEN** implementation MUST 将该行为迁到 owner CLI 或 owner module mode
- **AND** 删除旧 wrapper 时 MUST 不新增 alias、compat wrapper 或 fallback console script

#### Scenario: consolidated help 替代旧入口
- **WHEN** 旧 wrapper 被删除
- **THEN** `--help`、README、docs、OpenSpec current specs 和 inventory MUST 指向 consolidated owner command
- **AND** CLI help tests MUST 覆盖用户需要的新 mode 或 subcommand

### Requirement: 薄 wrapper 保留必须有 retained-with-reason
若某个诊断 wrapper 不能合并，项目 MUST 在 inventory 或 current spec 中记录保留理由、独立契约和删除触发条件。

#### Scenario: wrapper 仍承载独立契约
- **WHEN** wrapper 拥有独立输出 schema、claim evidence role、外部复现实验契约或不同 failure semantics
- **THEN** implementation MAY 保留该 wrapper
- **AND** retained-with-reason MUST 指明为什么 owner CLI mode 不足以替代它
