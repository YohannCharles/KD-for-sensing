## ADDED Requirements

### Requirement: Predictive Robustness artifact planning 必须可独立验证
Predictive JEPA robustness 的 condition rows、regional summaries、margin files、warnings、GPS-query advantage outputs、claim gate 和 diagnostics bundle MUST 由窄 helper 生成，并通过不需要真实 checkpoint 的测试覆盖。

#### Scenario: predictive artifacts 字段兼容
- **WHEN** manifest includes predictive robustness suites
- **THEN** predictive CSV/JSON 输出 MUST 保留 summary、margins、warnings、advantage metrics、claim gate 和 diagnostics bundle 的既有 key
- **AND** mock/smoke rows MUST remain clearly marked and MUST NOT become real numeric claims
