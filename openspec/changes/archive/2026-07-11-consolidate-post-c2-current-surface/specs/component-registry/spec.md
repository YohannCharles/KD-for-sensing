## ADDED Requirements

### Requirement: Retired zero-consumer components 不保留 registry surface
Geometry prior、safe reranker、GPS-query/predictive downstream、retired local baseline facade 和其它本 change 删除的 component MUST 不再注册为 current MODELS/HEADS/ENCODERS 名称。普通 unknown-name 错误 MUST 足以拒绝这些名称，不要求长期 removed-name table。

#### Scenario: Retired component 构建失败
- **WHEN** 用户请求本 change 删除的 geometry/query/legacy component 名称
- **THEN** registry MUST 拒绝构建
- **AND** 系统 MUST 不通过 alias 或 fallback 映射到 current component

#### Scenario: Current component 保持注册
- **WHEN** default component import 运行
- **THEN** U-Mask、current modular/fusion、MMW/CSI/physics 和 JEPA mean context owner MUST 继续可构建
- **AND** config import 轻量边界 MUST 保持
