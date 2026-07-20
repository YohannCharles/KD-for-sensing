## ADDED Requirements

### Requirement: AMBER-Full token padding 不得改变可用性语义
AMBER-Full 在对齐 modality spatial-token 维度时 MUST 为 padding token 生成不可用 mask；fusion attention、auxiliary loss 和 pooled diagnostics MUST 忽略 padding token。

#### Scenario: GPS token 少于 image spatial tokens
- **WHEN** image/radar/lidar 有多个 spatial token 而 GPS 只有一个 token
- **THEN** GPS 的补齐 token MUST 在 attention key-padding mask 中为 true
- **AND** GPS pooled feature MUST 只平均其真实 token
