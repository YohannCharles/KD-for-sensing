## ADDED Requirements

### Requirement: pattern-conditional BTAPA prototype loss
系统 MUST 支持 pattern-conditional BTAPA。启用 `use_pattern_conditional_btapa=true` 时，batch 内每个 sample MUST 根据 available mask 解析 pattern name；在 `btapa_apply_patterns` 中的样本 MUST 使用 BTAPA soft beam target，其它样本 MUST 在 `btapa_fallback_to_ordinary_proto=true` 时使用 ordinary prototype target。

#### Scenario: sample-wise 混合 target
- **WHEN** 同一 batch 同时包含 `radar_only` 和 `missing_gps`
- **THEN** `radar_only` 样本 MUST 使用 BTAPA soft beam target
- **AND** `missing_gps` 样本 MUST 使用 ordinary prototype target

#### Scenario: 缺失模态不参与 modality proto loss
- **WHEN** 某样本的 available mask 中 `radar=0`
- **THEN** radar modality feature MUST 不参与 modality prototype loss
- **AND** fusion feature MUST 继续参与 prototype loss

#### Scenario: diagnostics 记录 active ratio
- **WHEN** pattern-conditional BTAPA 启用
- **THEN** 训练 metrics MUST 记录 `ordinary_proto_loss`、`btapa_loss`、`btapa_active_ratio` 和 `total_proto_loss`
