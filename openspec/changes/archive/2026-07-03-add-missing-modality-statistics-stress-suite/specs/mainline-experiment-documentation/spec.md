## ADDED Requirements

### Requirement: Statistical and stress claim governance
主线实验文档 MUST 记录统计显著性和 stress suite 对 claim 升级的要求。单 seed、smoke、mock、not-comparable 或缺少 stress provenance 的结果 MUST 不写成正式论文结论。

#### Scenario: claim registry 记录统计证据
- **WHEN** 某缺失模态结果升级为 local strict-validation 或 local experimental claim
- **THEN** claim registry MUST 记录 seed_count、baseline、primary metric、mean/std 或 CI、comparability status、stress suite status 和 caveat
- **AND** 缺少任一必要证据时 claim status MUST 保持 pending、unverified、not_comparable 或 mock/smoke

#### Scenario: 实验矩阵区分 smoke 和 formal stress
- **WHEN** `docs/experiment_matrix.md` 或协议表列出 missing-modality stress suite
- **THEN** 文档 MUST 标明该 manifest 是 smoke、quick、formal、diagnostic-only 还是 evaluation-only
- **AND** 文档 MUST 指向 ignored 输出目录，不要求提交真实 stress metrics 或图表
