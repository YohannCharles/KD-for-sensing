## ADDED Requirements

### Requirement: Removed registry guard 修剪必须以迁移价值为准
组件注册表 SHALL 只为仍有当前迁移价值的退役名称保留 removed guard；低价值历史实现别名、旧实验变体和仅被 fixture 保活的名称 MUST 回落到普通 unknown-name 诊断或被删除。

#### Scenario: 有迁移价值的退役名称保留专门诊断
- **WHEN** 退役组件名称仍可能由当前用户配置、旧文档或受支持迁移路径触发，且存在明确替代组件或迁移方向
- **THEN** registry SHALL 保留 removed guard，并在错误信息中给出替代方向

#### Scenario: 低价值历史别名不再保活
- **WHEN** 退役名称只对应旧实现变体、历史强弱模型、teacher/student 临时类或不再支持的 feature extractor，且没有当前迁移路径
- **THEN** registry MUST 删除专门 removed guard；调用该名称时可以返回普通未知组件诊断

#### Scenario: 测试不得要求低价值 removed 文案
- **WHEN** 测试覆盖 registry 错误诊断
- **THEN** 测试 MUST 只断言 canonical 组件、仍保留的迁移 guard 和普通 unknown-name 行为，MUST NOT 通过大 fixture 要求所有历史名称都有专门 removed 文案
