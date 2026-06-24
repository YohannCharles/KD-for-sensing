## ADDED Requirements

### Requirement: GPS-query token readout evidence
系统 MUST 为 GPS-query K-token 输出路径提供可审计 readout evidence。该 evidence MUST 能区分 query token 是否形成稳定分工、readout 是否使用了非均匀 query 信息，以及 token 输出相对 frame 输出的 paired metric delta。该 evidence MUST 不改变默认 `GPSQueryPool` frame 输出语义。

#### Scenario: 记录 query diversity
- **WHEN** GPS-query pooler 输出或诊断包含 `[B,T,K,N]` attention map
- **THEN** 系统 MUST 计算 query-level attention entropy、effective patch count 和 query 间 diversity summary
- **AND** diagnostics MUST 记录 `k_queries`、token count、token grid、condition source 和 output mode
- **AND** diagnostics MUST detach attention map 后再用于日志、CSV、JSON 或图表

#### Scenario: 记录 token readout 权重
- **WHEN** downstream 使用 learned 或 attention-based token readout 消费 GPS-query tokens
- **THEN** 系统 MUST 在 runtime metadata 或 diagnostics 中记录 readout type、是否 trainable、readout weight summary 和输出 shape
- **AND** 系统 MUST 能把 learned readout 与 legacy uniform mean readout 区分开

#### Scenario: paired evidence 比较 frame 与 token 输出
- **WHEN** 系统生成 GPS-query 有效性或 architecture sweep evidence
- **THEN** evidence MUST 至少支持比较 `pooler_gps_query_k2_frame`、`pooler_gps_query_k2_tokens` 和新增 token readout candidate
- **AND** comparison MUST 使用相同 split、scene set、seed、checkpoint selection、metric profile 和 difficulty condition
- **AND** comparison MUST 输出 clean/P0、P1-P5 mean、Scene31、S31-S34 和 S32-S34 的 DBA 或等价主指标 delta

#### Scenario: 诊断不参与训练决策
- **WHEN** 模型 forward、loss 或 beam head 计算训练输出
- **THEN** query diversity、attention overlay、case label 和 benchmark condition id MUST NOT 作为模型输入
- **AND** 这些字段 MUST 只用于离线诊断、分组统计、claim gate 或报告
