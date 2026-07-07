## ADDED Requirements

### Requirement: Scene31 next-round summary 可由共享 owner 产出
Scene31 next-round 相关 summary MAY 与 BC-next、fresh eval、subset reliability、patternfilm、funnel 和 subset reference summary 共享一个参数化 Scene31 summary owner。共享 owner MUST 保持各 workflow 的默认输入、输出 schema、标签和错误信息可追溯。

#### Scenario: next-round summary 使用 profile 参数
- **WHEN** 协作者需要生成 Scene31 next-round summary
- **THEN** 推荐入口 MAY 是共享 Scene31 summary owner 加显式 profile、group 或 view 参数
- **AND** 项目 MUST 不为每个 profile 保留只设置默认参数的重复脚本

#### Scenario: 输出契约保持稳定
- **WHEN** implementation 将 next-round summary 迁入共享 owner
- **THEN** 旧 workflow 已承诺的 summary 字段、排序、筛选规则和默认 artifact 路径 MUST 保持稳定或同步更新 current spec
- **AND** focused tests MUST 覆盖共享 owner 的 next-round profile

### Requirement: Scene31 summary 删除必须防止脚本回流
删除 Scene31 next-round 周边重复 summary 脚本后，scripts inventory 或 architecture guardrail MUST 拒绝同职责 wrapper 回流，除非新的 OpenSpec reason 明确说明独立脚本的当前价值。

#### Scenario: guardrail 拦截重复 summary wrapper
- **WHEN** 新增脚本只调用共享 Scene31 summary owner 并设置固定 profile
- **THEN** surface guardrail SHOULD 报告该脚本需要删除或登记 retained-with-reason
- **AND** docs SHOULD 直接展示共享 owner 命令
