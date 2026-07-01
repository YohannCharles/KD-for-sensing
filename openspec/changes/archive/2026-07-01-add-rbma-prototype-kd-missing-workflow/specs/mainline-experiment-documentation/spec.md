## ADDED Requirements

### Requirement: RBMA ablation documentation
主线实验文档 MUST 记录 RBMA/prototype/KD missing-modality ablation 的 local/pending status、配置入口、推荐运行顺序、比较口径和 claim caveat。文档 MUST 不把未验证本地实验描述为 official reproduction 或已达成数值 claim。

#### Scenario: 文档记录推荐四配置
- **WHEN** RBMA ablation configs 加入仓库
- **THEN** `docs/experiment_matrix.md` 或等价 current 文档 MUST 记录首轮推荐运行 `amber_style_mask_baseline`、`no_jepa_rbma`、`no_jepa_rbma_proto` 和 `no_jepa_rbma_proto_kd`
- **AND** 文档 MUST 说明 `jepa_small_lambda_rbma_proto_kd` 是后续对照而非首轮必跑项

#### Scenario: claim registry 保持 pending/local
- **WHEN** 文档记录 RBMA workflow 结果入口或实验计划
- **THEN** `docs/result_claims_registry.md` 或等价 claim 账本 MUST 将其标记为 local/pending，直到真实评估结果和 provenance 完整
- **AND** 文档 MUST 不声称 AMBER official 数值复现已完成

#### Scenario: 实验协议记录 pattern 口径
- **WHEN** 文档描述 missing pattern evaluation
- **THEN** `docs/experiment_protocols.md` 或等价协议文档 MUST 记录 canonical 模态顺序、pattern definitions、pattern probabilities、hard-label metrics 和输出边界
- **AND** 文档 MUST 明确内部使用 `image` 而不是 `vision` 作为 canonical 模态名
