## REMOVED Requirements

### Requirement: 静态 HTML evidence dashboard
**Reason**: HTML renderer 只服务已退役 research dashboard，没有独立 current consumer。
**Migration**: 使用正式 Markdown/CSV claim docs、run index JSON 和 paper export artifacts。

#### Scenario: HTML renderer 删除
- **WHEN** research dashboard 退役
- **THEN** 静态 HTML evidence renderer MUST 不再属于 current source
- **AND** 不新增替代 Web 或静态 dashboard

### Requirement: HTML dashboard 输出边界
**Reason**: Dashboard 输出能力随 renderer 删除。
**Migration**: Retained owners 继续写入各自 ignored output root。

#### Scenario: 不再写 dashboard HTML
- **WHEN** current diagnostics 运行
- **THEN** 系统 MUST 不要求 dashboard `.html` 输出
- **AND** paper/analysis owners 的输出边界保持不变

### Requirement: 离线安全渲染
**Reason**: 不再渲染动态 dashboard HTML，因此无需维护 escaping/CDN 契约。
**Migration**: 其它 HTML 输出若未来出现，必须在其 owner change 中重新定义安全边界。

#### Scenario: 删除 renderer 后无网络依赖
- **WHEN** current CLI 运行
- **THEN** 它们 MUST 不加载已删除 dashboard renderer
- **AND** 不要求 dashboard HTML security tests

### Requirement: HTML dashboard 验证
**Reason**: Capability 整体退役。
**Migration**: 删除 renderer/CLI 专属测试，保留其它 owner focused tests。

#### Scenario: Dashboard tests 删除
- **WHEN** implementation 删除 dashboard source
- **THEN** 专属 renderer/CLI tests MUST 删除
- **AND** CLI smoke MUST 不再要求 HTML 参数
