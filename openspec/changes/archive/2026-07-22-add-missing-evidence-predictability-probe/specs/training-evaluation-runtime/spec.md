## ADDED Requirements

### Requirement: Frozen-cache recovery probe 必须保持 inner-validation 选择边界
runtime MUST 允许 missing-evidence probe从审计过的 C0 clean cache训练独立轻量模型，且 MUST 不加载或更新 semantic backbone、prototype bank或 C0 checkpoint参数。checkpoint MUST 只由 inner-validation recovery objective选择；最终 beam、weather、sector、outer test和oracle-gap结果不得用于 checkpoint选择或 normalization拟合。

#### Scenario: 选择 recovery probe checkpoint
- **WHEN** Linear或MLP evidence/residual完成多个 cache epoch
- **THEN** best checkpoint MUST 对应最低有限 inner-validation recovery loss
- **AND** checkpoint provenance MUST 记录 missing modality、probe类型、C0 SHA、cache manifest SHA、train normalization、seed和 claim-ineligible状态
- **AND** checkpoint MUST 只包含 probe参数和审计metadata

### Requirement: Missing-evidence probe 运行产物必须保持本地边界
runtime MUST 将 probe manifest、状态、checkpoint、CSV、图表和总结写入 ignored `outputs/missing_evidence_probe/`，并 MUST 标记 single-seed、inner-only、outer-test=false和 claim-eligible=false。

#### Scenario: 完成离线可行性验证
- **WHEN** 四个 missing方向完成汇总
- **THEN** launcher MUST 退出且不修改 canonical recipe或正式 claim
- **AND** 下一轮 fallback adapter、multi-seed或 outer-test实验 MUST 只能记录为建议而不得自动执行
