## ADDED Requirements

### Requirement: Registry helper surface 必须最小化
组件注册表 MUST 只暴露构建、查询、注册和错误诊断所需 API。无当前调用方、无 CLI、无 docs current 消费、无测试必要性的自检 helper MUST 删除，而不是作为 public API 长期保留。

#### Scenario: 删除 registry self check
- **WHEN** `registry_self_check` 没有项目内调用方且 registry 行为已由 focused tests 覆盖
- **THEN** 本 change MUST 删除该 helper 和 `__all__` 导出
- **AND** component registry tests MUST 继续覆盖 build、unknown name、duplicate name 和 missing required parameter 错误

#### Scenario: 不新增替代 smoke helper
- **WHEN** registry self check 被删除
- **THEN** 项目 MUST 不新增等价的长期 smoke function 或 CLI
- **AND** 必要验证 MUST 留在 pytest focused tests 中

### Requirement: Removed guard 表只保留当前迁移价值
注册表和配置 guard MAY 为仍常见或仍有当前迁移路径的旧名称提供专属 removed error。完全退役且已由 OpenSpec tombstone、inventory 和 README retired wording 覆盖的历史路线 MUST 不要求每个 registry 或 facade 继续维护专属 removed guard。

#### Scenario: 保留高频迁移 guard
- **WHEN** 用户请求 `scenario31` dataset alias、KD loss token、removed image profile 或 removed image encoder
- **THEN** 系统 SHOULD 继续给出清晰迁移错误
- **AND** 错误 MUST 指向当前 canonical dataset、loss 或 image profile 入口

#### Scenario: 低价值 retired 名称回落 unknown
- **WHEN** 用户请求完全退役且不再有当前迁移目标的旧研究线 registry 名称
- **THEN** 系统 MAY 返回普通 unknown-name registry 错误
- **AND** 系统 MUST 不通过 deprecated alias、facade 或 virtual config 重定向到当前实现

### Requirement: Registry 公开导出不得镜像非 API 细节
`__all__` MUST 只包含真实 public API。删除 helper、removed alias 或内部 registry 表时，`__all__` MUST 同步收缩；项目 MUST 不为了保持旧导出而保留空 wrapper。

#### Scenario: 删除导出后 import 失败
- **WHEN** 非推荐外部代码从 registry 模块导入已删除 helper
- **THEN** 导入 MAY 失败
- **AND** 当前构建流程和 focused tests MUST 不依赖该 helper
