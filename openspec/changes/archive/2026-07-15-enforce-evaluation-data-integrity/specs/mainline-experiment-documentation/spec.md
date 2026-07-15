## ADDED Requirements

### Requirement: 泄漏影响的 temporal evidence 必须降级
主线 claim、protocol 和 history 文档 MUST 将使用逐样本拆分重叠 temporal window 或 test-as-validation 的结果标为 `not_comparable` 或 `invalidated`。此类结果 MUST NOT 进入 reviewed main claim 或 paper main table。

#### Scenario: H5/P1 旧 split evidence
- **WHEN** 文档引用修复前 H5/P1 temporal matrix 结果
- **THEN** claim status MUST 标记为 `not_comparable`
- **AND** caveat MUST 记录 sequence group 与历史/target 帧跨 split 泄漏
- **AND** 文档 MUST 不把旧数值用于方法优劣结论

#### Scenario: Temporal evidence 重新晋级
- **WHEN** 新 H5/P1 结果请求升级为 local strict-validation 或 reviewed claim
- **THEN** provenance MUST 包含 group-safe split artifact、sample/frame identity audit、独立 validation、final test、seed 和 normalization fingerprint
- **AND** 任一字段缺失时 status MUST 保持 pending、unverified 或 not_comparable
