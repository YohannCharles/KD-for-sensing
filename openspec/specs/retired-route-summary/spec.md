# retired-route-summary Specification

## Purpose

规定退役路线的唯一历史记录方式，确保历史用途可由集中说明和 archive 追溯，但不保留运行代码、配置、命令、alias 或迁移层。

## Requirements

### Requirement: 折叠退役路线不属于 current support surface

除 T2、S1、AMBER-Full、RMBP-MM 及其 MMW 运行闭包外，历史 source、config、CLI、test、registry entry、alias、migration guard 与 local runbook MUST 不属于 current surface。

#### Scenario: 用户查询退役能力

- **WHEN** 维护者需要了解某条已删除路线
- **THEN** `docs/retired_routes.md` MUST 给出简短用途和 archive/git history 追溯方式
- **AND** 系统 MUST 不要求旧模块、配置或命令仍存在

### Requirement: 历史说明不构成兼容承诺

历史说明 MUST 不提供可运行命令、YAML 映射或 compatibility stub。

#### Scenario: 请求退役路径

- **WHEN** 用户加载旧配置或导入旧模块
- **THEN** 普通缺失文件或 unknown-name 错误即可
- **AND** 系统 MUST 不自动迁移或构建替代路径
