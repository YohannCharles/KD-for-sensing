## REMOVED Requirements

### Requirement: GPS-query Attention Pooling 模块
**Reason**: GPS-query pooler 没有 current tracked config、CLI 或 claim consumer，current JEPA downstream 只保留 mean pooling。
**Migration**: 删除 pooler registry/implementation；未来恢复须新建 OpenSpec change。

#### Scenario: GPS-query pooler 不再构建
- **WHEN** config 请求 GPS-query attention pooling
- **THEN** component construction MUST fail clearly
- **AND** 系统 MUST 不静默回退到 mean pooling

### Requirement: JEPA context image GPS-query pooling
**Reason**: `jepa_context_image` 的 GPS-conditioned readout 已退出 current support surface。
**Migration**: Current JEPA context reuse 使用 patch-token mean pooling，并继续加载 current checkpoint。

#### Scenario: Context encoder 只承诺 mean pooling
- **WHEN** current `jepa_context_image` 被构建
- **THEN** 它 MUST 不要求 GPS-query pooling path
- **AND** mean pooling output contract MUST 保持可用

### Requirement: fair_gps_biased 派生配置
**Reason**: 该派生配置只服务已退役 GPS-query 对照实验。
**Migration**: 删除 config 与运行入口；历史比较从 archive 查询。

#### Scenario: 派生配置退出
- **WHEN** 用户请求 `fair_gps_biased` 或等价旧配置
- **THEN** config loader MUST 返回 unknown/removed failure

### Requirement: GPS-query pooling metadata
**Reason**: Pooler 删除后，k-query、condition source 与 attention metadata 没有 producer。
**Migration**: Current mean path 只保留 checkpoint、encoder type 与 pooling mode metadata。

#### Scenario: Query metadata 不再输出
- **WHEN** current JEPA mean encoder 写出 metadata
- **THEN** 系统 MUST 不要求 GPS-query pooling metadata fields

### Requirement: GPS-query token readout evidence
**Reason**: Token-read evidence 只服务已退役 query-pooling sweep 与 claim。
**Migration**: 删除专属 evidence producer；历史 evidence 从 archive/ignored artifacts 查询。

#### Scenario: Token-read evidence 退出
- **WHEN** current evaluation 或 summary 运行
- **THEN** 系统 MUST 不要求 GPS-query token readout evidence

