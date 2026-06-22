## MODIFIED Requirements

### Requirement: CSI hardening sweep 分析脚本
系统 SHOULD 保留 CSI hardening debug/sweep 的解释边界和关键判定阈值，但 MUST 不要求长期维护一次性 `scripts/analyze_csi_hardening_sweep.py` 分析脚本。需要复查 sweep 时，开发者 SHOULD 使用 run history、resolved config、debug matrix parity 和 `docs/research_notes.md` 中的 high-ceiling/slow-to-learn 判定阈值。

#### Scenario: 历史分析脚本可退役
- **WHEN** 当前 workflow 不再需要 `scripts/analyze_csi_hardening_sweep.py`
- **THEN** 项目 MAY 删除该脚本和只服务它的测试
- **AND** CSI hardening 配置、训练脚本和文档 MUST 不继续要求该脚本存在

#### Scenario: 解释边界保留
- **WHEN** 开发者解释 CSI hardening sweep 或 debug matrix
- **THEN** 文档 MUST 保留 destructive negative control、high-ceiling/slow-to-learn、clone parity 和 debug-first caveat
- **AND** 结论 MUST 不把未验证本地 run 写成正式结果 claim
