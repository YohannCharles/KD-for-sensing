## ADDED Requirements

### Requirement: Target-shot split runtime metadata
Dataset runtime metadata MUST record target-shot split state when a run or diagnostic consumes a target-shot split artifact. Metadata MUST include source domains, target domains, target_label_fraction, target_labeled sample count, target_unlabeled sample count, target_test sample count, split artifact path, seed and strict eligibility summary.

#### Scenario: runtime 记录 target-shot split
- **WHEN** 训练、适配、评估或诊断构建 dataloader 并传入 target-shot split artifact
- **THEN** runtime metadata MUST 记录 source/target domain、target_label_fraction、各 split 样本数和 artifact 路径
- **AND** metadata MUST 记录 split strict eligibility 或 leakage diagnostics 摘要

### Requirement: Geometry-residual target schema metadata
Dataset runtime metadata MUST distinguish absolute beam target schema from geometry-residual target schema. When geometry-residual labels are enabled, metadata MUST record num_beams, beam_geo source, residual convention, max_residual, overflow strategy and num_geo_sectors.

#### Scenario: runtime 记录 geometry-residual schema
- **WHEN** dataset 使用 `label_space.type: geometry_residual`
- **THEN** runtime metadata MUST 记录当前 target schema 为 geometry-residual
- **AND** metadata MUST 包含 residual convention、max_residual 和 geometry availability summary

### Requirement: Labeled 与 unlabeled target subset guard
训练 runtime MUST 区分 `target_labeled` 与 `target_unlabeled` subset。`target_unlabeled` batch 的 beam、residual、beam_power、CSI/channel、path 和 radio supervision 字段 MUST 受到 sensitive field guard 保护；`target_labeled` batch MAY 使用 beam/residual supervision，但仍 MUST 遵守 path/radio opt-in policy。

#### Scenario: unlabeled residual supervision 被拒绝
- **WHEN** adaptation loss 在 `target_unlabeled` batch 上读取 `beam_residual` 或 `residual_class` 作为监督
- **THEN** runtime guard MUST raise error
- **AND** error message MUST 包含 subset、field name 和 split artifact path

#### Scenario: labeled target residual supervision 被记录
- **WHEN** adaptation loss 在 `target_labeled` batch 上读取 residual supervision
- **THEN** run metadata MUST 记录 `used_target_residual_for_training=true`
- **AND** metadata MUST 表明监督来源仅限 target_labeled subset
