## ADDED Requirements

### Requirement: 主实验证据收敛记录
主线实验文档 MUST 在主实验进入证据收敛阶段时记录 final checklist、缺失 evidence、claim status 和下一步最小动作。文档 MUST 区分“继续补证据”和“新增方法搜索”，并在主方法冻结时说明冻结边界。

#### Scenario: Scene31-34 evidence 更新
- **WHEN** Scene31-34 final summary、paper tables 或 claim status 发生变化
- **THEN** `docs/mainline_experiment_history.md`、`docs/mainline_model_catalog.md`、`docs/experiment_protocols.md` 或 `docs/result_claims_registry.md` 中的对应 current fact MUST 同步更新
- **AND** 真实 metrics、figures、logs 和 checkpoint MUST 继续留在 ignored output root

#### Scenario: JEPA benchmark 从 smoke 转 real
- **WHEN** JEPA shortcut 或 predictive robustness benchmark 从 smoke manifest 转为 real manifest
- **THEN** 文档 MUST 更新 claim status、manifest provenance 和 caveat
- **AND** synthetic smoke 结果 MUST 继续标记为 mock/smoke
