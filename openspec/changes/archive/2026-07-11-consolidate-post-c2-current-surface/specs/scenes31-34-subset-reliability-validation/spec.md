## REMOVED Requirements

### Requirement: Scene31-34 subset reliability runner
**Reason**: subset-reliability quick validation 是已冻结的 local/manual 支线，不属于保留的 Scene31-34 final evidence workflow。
**Migration**: Current multi-scene missing-modality validation 使用 `scenes31-34-main-missing-modality-workflow`、U-Mask eval matrix 和 final C2 evidence owner；历史 seed 结果只读保留。

#### Scenario: subset-reliability runner 退出
- **WHEN** current scripts 和 inventory 被枚举
- **THEN** 项目 MUST 不要求 `run_scenes31_34_subset_reliability` runner、专属 group 或 configs 存在
- **AND** Scene31-34 main runner、fresh eval 和 final analysis MUST 保持

### Requirement: Scene31-34 summary outputs
**Reason**: 该 summary schema 只服务已退役 subset-reliability 支线，与 protected Scene31-34 final analysis 的 pooled/per-scene evidence 重复。
**Migration**: Current pooled、per-scene、stability 和 paper-facing outputs 以 `scenes31-34-main-missing-modality-workflow` 为权威；历史 subset summary 从 archive/git 查询。

#### Scenario: subset summary 不再生成
- **WHEN** current reporting surface 被检查
- **THEN** 系统 MUST 不要求旧 subset-reliability summary 脚本或其专属 CSV/Markdown 文件
- **AND** protected final analysis outputs 和 claim provenance MUST 不被删除或降级
