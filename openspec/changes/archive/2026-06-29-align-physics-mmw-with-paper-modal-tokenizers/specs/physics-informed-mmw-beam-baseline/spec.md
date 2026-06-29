## ADDED Requirements

### Requirement: Paper-style modality tokenizer frontend
`pinn_multimodal_beam` SHALL support a configurable paper-style frontend that maps enabled modalities into unified latent tokens before physics-aware prediction. The frontend MUST support modality-specific tokenizers, modality embedding, time/position embedding, a shared Transformer fusion block, and a deterministic pooling or horizon adapter that returns `[B, num_pred, hidden_dim]` latent states for the existing direct beam head and path parameter head.

#### Scenario: 多模态 token 进入共享 Transformer
- **WHEN** `model.primary.type=pinn_multimodal_beam` 且配置启用 paper-style tokenizer frontend
- **THEN** image、radar、lidar、gps、mmwave 或 csi 输入 MUST 先被各自 tokenizer 映射为统一 `hidden_dim` token
- **AND** fusion MUST 在 token 上加入 modality embedding 和 time/position embedding
- **AND** 模型输出 MUST 继续包含 `logits`、`direct_logits`、`physics_logits`、`path_hat`、`h_hat`、`latent` 和 `shape_metadata`

#### Scenario: 单模态输入仍可运行
- **WHEN** paper-style tokenizer frontend 只启用一个模态
- **THEN** 模型 MUST 只要求该模态输入
- **AND** shared Transformer MUST 仍返回可供 direct head 和 path head 消费的 `[B, num_pred, hidden_dim]` latent

### Requirement: JEPA context image tokenizer without GPS context
Image tokenizer for the paper-style physics MMW baseline MUST reuse the existing `jepa_context_image` encoder. Formal experiment configs MUST provide a pretrained checkpoint path or explicitly mark the run as non-formal debug/smoke. The image tokenizer MUST NOT require or consume GPS context, GPS query pooling, or `gps_condition_features`.

#### Scenario: 图像 tokenizer 使用 JEPA context encoder
- **WHEN** paper-style physics MMW 配置启用 image 模态
- **THEN** `model.primary.encoders.image.type` 或等价 tokenizer 配置 MUST 解析到 `jepa_context_image`
- **AND** pooling MUST 使用 `mean` 或其它不声明 GPS context 依赖的 pooler
- **AND** 模型 forward MUST NOT 向图像 encoder 传入 `gps_condition_features`

#### Scenario: 正式实验要求预训练 checkpoint
- **WHEN** paper-style physics MMW 配置未提供 `jepa_context_image` checkpoint
- **THEN** run metadata MUST 标记 `formal_experiment_eligible=false`
- **AND** debug/smoke 以外的正式实验配置 MUST fail fast 或给出清晰配置错误

### Requirement: Restricted wireless input for physics chain
Paper-style physics MMW training MUST distinguish restricted wireless observation from full CSI supervision. Full current narrowband array CSI MUST remain `physics_targets.csi_target` by default; model input MUST use `csi_input` or equivalent restricted observation such as sparse pilot, sparse antenna observation, low-dimensional RF scan, or compressed latent. `oracle_full` MUST remain an explicitly authorized upper-bound mode.

#### Scenario: sparse pilot CSI 输入与完整 CSI 监督分离
- **WHEN** `data.use_csi_input=true` 且 `data.csi_input_mode=sparse_pilot`
- **THEN** model input MUST receive `csi_input` and MAY receive `csi_observation_mask`
- **AND** `physics_targets.csi_target` MUST NOT be passed to model forward
- **AND** reconstruction loss MUST compare `h_hat` against `physics_targets.csi_target` using shape metadata aligned to `[B, T, Nsc, Nant, 2]`

#### Scenario: oracle full CSI 只作为上界
- **WHEN** `data.csi_input_mode=oracle_full`
- **THEN** 系统 MUST require `data.allow_oracle_full_csi_input=true`
- **AND** run metadata MUST 标记 `oracle_upper_bound=true` 和 `main_conclusion_eligible=false`

### Requirement: Narrowband array-channel claim boundary
The physics-informed MMW baseline MUST describe its reconstruction target as narrowband array-channel reconstruction unless the dataset and configuration provide multiple observed subcarriers. The system MUST NOT label `[T, 1, Nant, 2]` supervision as complete wideband CSI reconstruction.

#### Scenario: 单子载波 CSI target 的 metadata
- **WHEN** `physics_targets.csi_target` shape has `Nsc=1`
- **THEN** model or run metadata MUST record `channel_target_scope=narrowband_array_channel`
- **AND** reports MUST NOT claim complete wideband CSI reconstruction
