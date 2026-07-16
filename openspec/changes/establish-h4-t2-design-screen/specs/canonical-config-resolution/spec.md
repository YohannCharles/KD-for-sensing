## ADDED Requirements

### Requirement: U-Mask profile 解析稳定且不污染 canonical base recipe
tracked MMW T2/S1 base recipe 与 tracked mainline launcher MUST 在没有 outputs、checkpoint 或数据的干净 clone 中解析 H4 U-Mask training fields。legacy H0 只能由 tracked launcher 的显式 selector 写入 generated configuration；H4 和 H0 selector MUST 不依赖 output YAML 或历史 resolved config，且 H4 不得隐式改写 legacy protocol。

#### Scenario: 生成 H4 mainline config
- **WHEN** 用户从 tracked mainline launcher 请求 `umask_h4_v1`
- **THEN** generated config MUST 解析为完整 H4 optimizer/scheduler fields
- **AND** 不得读取本地训练产物以确定 profile
