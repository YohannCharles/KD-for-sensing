## MODIFIED Requirements

### Requirement: Predictive Robustness 主场景
系统 MUST 提供 Predictive Robustness benchmark suite，用于评估当前图像不可观测、图像受干扰或 GPS 受干扰时，JEPA 是否能利用历史视觉上下文和预测 latent 保持 beam prediction 性能。该 suite MUST 使用 clean anchor 加少数单轴 stress curves 作为主评估证据；融合机制区分 MUST 结合正交 CxD/A-slice 诊断证据；现有 C0-C4 x D0-D7 matrix MUST 继续作为 overall sanity。旧 P0-P5 离散条件 MAY 作为 legacy/deprecated 输入被读取，但 MUST NOT 作为默认主 claim 口径。

#### Scenario: 解析 canonical predictive robustness stress suite
- **WHEN** benchmark manifest 引用 `predictive_jepa_robustness` canonical preset
- **THEN** 系统 MUST 至少支持 clean anchor、`image_missing`、`image_noise` 和 `gps_noise` 三条默认 stress curves
- **AND** 每条 stress curve MUST 声明 severity values、severity unit、operator 参数、history window、seed、split、difficulty digest 和 replay metadata
- **AND** 所有 stress condition MUST 保持 `target_beam`、`beam_power`、sample id 和 split metadata 不变

#### Scenario: 低区分度 P-level 不进入默认主评估
- **WHEN** manifest 未显式请求 legacy P0-P5 preset
- **THEN** 系统 MUST NOT 默认生成 `P1_current_frame_missing_history_available`、`P2_semantic_occlusion_history_available` 或 `P5_novel_weather_history_available` 行
- **AND** 系统 MUST NOT 将旧 `P4_joint_predictive_recovery` 表达为当前图像缺失后再叠加随机遮挡

#### Scenario: 单轴 stress 不混合语义
- **WHEN** 系统生成 `image_missing` stress curve
- **THEN** 该 curve MUST 只表达图像缺失强度，并 MUST NOT 同时启用遮挡、天气或 wrong GPS
- **AND** `image_noise` MUST 只表达图像干扰强度，并 MUST 保持 GPS 输入不变
- **AND** `gps_noise` MUST 只表达 GPS 干扰强度，并 MUST 保持 image 输入不变

#### Scenario: P-suite 不单独替代融合机制诊断
- **WHEN** predictive robustness benchmark 完成
- **THEN** 输出 MUST 同时记录 stress curve metrics 和可选 CxD/A-slice diagnostic metrics
- **AND** report MUST 明确 stress curve summary 不得单独作为融合机制区分的主证据
- **AND** 当 reused-weight fusion diagnostic metrics 可用时，report MUST 优先使用正交 CxD/A-slice 指标解释融合行为

### Requirement: 5 个百分点 claim 口径
系统 MUST 为 predictive robustness 主 claim 提供严格可比较的 margin-vs-ResNet 口径。只有在 split、sample_count、label space、metric profile、difficulty digest 和 seed 可比时，系统 MAY 将 `jepa_predictive_hybrid` 相对 Image ResNet+GPS 的 stress-curve robustness margin 标记为主 claim；融合机制 claim MUST 另行满足正交 CxD/A-slice 诊断证据。

#### Scenario: 计算 stress robustness margin
- **WHEN** benchmark 同时包含 `jepa_predictive_hybrid` 与 Image ResNet+GPS strict comparable stress rows
- **THEN** 系统 MUST 计算每个 stress suite 的 clean metric、stress metric、retention、AUC retention、collapse severity 和 margin-vs-ResNet
- **AND** `claim_pass_5pt` MUST 仅在 configured primary stress summary 的 `margin_vs_resnet_dba >= 0.05` 且 comparability status 为 strict 时为 true

#### Scenario: Stress claim 与融合诊断 claim 分离
- **WHEN** stress-curve margin 达到阈值但 reused-weight CxD/A-slice diagnostic metrics 缺失或 not-comparable
- **THEN** 系统 MAY 标记 predictive robustness stress claim
- **AND** 系统 MUST 不将该结果表述为融合机制已被正交诊断验证

#### Scenario: smoke 不升级为真实 claim
- **WHEN** benchmark 使用 synthetic metrics、mock weights、partial model set 或 missing checkpoint
- **THEN** 系统 MUST 将 claim status 标记为 `mock/smoke`、`pending` 或 `unavailable`
- **AND** docs 和 result claims registry MUST 不记录真实性能数值

### Requirement: 训练 profile 与完整 stress benchmark 分离
Predictive Robustness MUST 区分训练 difficulty profile、evaluation difficulty profile 和 stress-curve benchmark suite。单个训练 profile、单个 clean evaluation profile 或 legacy P-level profile MUST NOT 被描述为完整 stress benchmark。

#### Scenario: 训练配置只启用部分 stress condition
- **WHEN** 派生训练配置只声明单个 missing、noise、wrong-GPS 或 legacy P-level condition
- **THEN** 文档 MUST 将其描述为训练/curriculum profile
- **AND** 文档 MUST 指向 benchmark manifest 或本地 real manifest 才能执行完整 stress-curve evaluation

#### Scenario: 完整 benchmark claim 需要 stress provenance
- **WHEN** claim registry 准备将 Predictive Robustness 升级为真实 claim
- **THEN** provenance MUST 包含 clean anchor、每条默认 stress curve 的 condition-level metrics、strict comparability fields、difficulty digest、seed、split、sample_count 和 Image ResNet+GPS baseline
- **AND** 缺少任一 required provenance 时 claim status MUST 保持 `pending`、`unavailable` 或 `not_comparable`

### Requirement: GPS-query++ claim gate
Predictive GPS-query++ claim status MUST require canonical stress-curve evidence and MAY use advantage-slice evidence as mechanism support. Advantage-slice improvements MAY explain mechanism but MUST NOT alone promote a claim.

#### Scenario: Claim gate 计算
- **WHEN** stress curves and advantage slice metrics are available for strict comparable models
- **THEN** system MUST compute stress robustness margin vs Image ResNet+GPS、advantage-slice margin vs Image ResNet+GPS、advantage-slice margin vs current GPS-query baseline and claim pass flags
- **AND** primary claim pass MUST remain based on canonical predictive robustness stress criteria and configured margin threshold

#### Scenario: Advantage slice 单独提升不升级 claim
- **WHEN** Predictive GPS-query++ outperforms baselines on advantage slice but not on canonical stress curves
- **THEN** report MUST describe the result as mechanism evidence or targeted advantage
- **AND** claim status MUST remain pending, partial, unavailable or not-comparable according to provenance
