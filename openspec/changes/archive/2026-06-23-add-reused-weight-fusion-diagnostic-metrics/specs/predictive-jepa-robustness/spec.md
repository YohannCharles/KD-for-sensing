## MODIFIED Requirements

### Requirement: Predictive Robustness 主场景
系统 MUST 提供 Predictive Robustness benchmark suite，用于评估当前图像不可观测或 GPS 可信但错误时，JEPA 是否能利用历史视觉上下文和预测 latent 保持 beam prediction 性能。该 suite MUST 保留 P0-P5 condition-level benchmark 作为兼容鲁棒性证据，但融合机制区分 MUST 结合正交 CxD/A-slice 诊断证据；现有 C0-C4 x D0-D7 matrix MUST 继续作为 overall sanity。

#### Scenario: 解析 canonical predictive robustness suite
- **WHEN** benchmark manifest 引用 `predictive_jepa_robustness` canonical preset
- **THEN** 系统 MUST 至少支持 `P0_clean_current`、`P1_current_frame_missing_history_available`、`P2_semantic_occlusion_history_available`、`P3_plausible_wrong_gps_current_image`、`P4_joint_predictive_recovery` 和 `P5_novel_weather_history_available`
- **AND** 每个 condition MUST 记录 image/GPS operator 参数、history window、seed、split、difficulty digest 和 replay metadata
- **AND** 所有 condition MUST 保持 `target_beam`、`beam_power`、sample id 和 split metadata 不变

#### Scenario: P-suite 不单独替代融合机制诊断
- **WHEN** predictive robustness benchmark 完成
- **THEN** 输出 MUST 同时记录 P0-P5 regional metrics 和可选 CxD/A-slice diagnostic metrics
- **AND** report MUST 明确 P0-P5 mean 不得单独作为融合机制区分的主证据
- **AND** 当 reused-weight fusion diagnostic metrics 可用时，report MUST 优先使用正交 CxD/A-slice 指标解释融合行为

### Requirement: 5 个百分点 claim 口径
系统 MUST 为 predictive robustness 主 claim 提供严格可比较的 margin-vs-ResNet 口径。只有在 split、sample_count、label space、metric profile、difficulty digest 和 seed 可比时，系统 MAY 将 `jepa_predictive_hybrid` 相对 Image ResNet+GPS 的 predictive DBA margin 标记为 P-suite claim；融合机制 claim MUST 另行满足正交 CxD/A-slice 诊断证据。

#### Scenario: 计算 predictive DBA margin
- **WHEN** benchmark 同时包含 `jepa_predictive_hybrid` 与 Image ResNet+GPS strict comparable rows
- **THEN** 系统 MUST 计算 `predictive_dba`、`predictive_top1`、`resnet_predictive_dba`、`margin_vs_resnet_dba` 和 `claim_pass_5pt`
- **AND** `claim_pass_5pt` MUST 仅在 `margin_vs_resnet_dba >= 0.05` 且 comparability status 为 strict 时为 true

#### Scenario: P-suite claim 与融合诊断 claim 分离
- **WHEN** P0-P5 margin 达到阈值但 reused-weight CxD/A-slice diagnostic metrics 缺失或 not-comparable
- **THEN** 系统 MAY 标记 P-suite robustness claim
- **AND** 系统 MUST 不将该结果表述为融合机制已被正交诊断验证

#### Scenario: smoke 不升级为真实 claim
- **WHEN** benchmark 使用 synthetic metrics、mock weights、partial model set 或 missing checkpoint
- **THEN** 系统 MUST 将 claim status 标记为 `mock/smoke`、`pending` 或 `unavailable`
- **AND** docs 和 result claims registry MUST 不记录真实性能数值
