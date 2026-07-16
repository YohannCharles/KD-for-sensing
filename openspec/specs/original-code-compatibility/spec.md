# original-code-compatibility Specification

## Purpose

明确 current MMW T2/baseline 与受限 DeepSense6G T2 surface 不保留旧代码兼容层，并让 checkpoint resume、配置加载和 package import 在遇到过时输入时以普通、明确的错误失败。

## Requirements

### Requirement: 恢复训练

current T2/baseline resume MUST 只接受与当前 MMW 或受限 DeepSense6G recipe 身份匹配的 current-schema checkpoint，并在 checkpoint 不存在或必要状态缺失时 fail closed；系统 MUST 不用旧字段、旧路径或 alias 伪造恢复。

#### Scenario: current checkpoint resume

- **WHEN** 用户请求从 current `last.pth` 恢复
- **THEN** runtime MUST 验证 checkpoint 与当前 recipe 的必要身份
- **AND** 不得覆盖源 run 的产物

#### Scenario: 无效或历史 checkpoint

- **WHEN** 路径不存在或 checkpoint 不符合 current schema
- **THEN** runtime MUST 在训练开始前报告明确错误
- **AND** 不得回退到 fresh run、迁移分支或兼容 alias

### Requirement: 旧代码仅保留历史说明

项目 MUST 不提供旧模块、命令或 checkpoint schema 的 compatibility facade；其历史作用只可由 `docs/retired_routes.md` 或 archive 说明。

#### Scenario: 请求退役输入

- **WHEN** 用户请求旧模块、旧命令或过时 checkpoint schema
- **THEN** 系统 MUST 返回普通缺失或 current-schema 错误
- **AND** 不得加载 alias、facade 或迁移路径
