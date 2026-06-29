# physics-informed-mmw-beam-baseline Specification

## Purpose
定义 physics-informed MMW beam baseline 的数据边界、受限 CSI 输入、路径/阵列物理链路、paper-style 多模态 tokenizer 前端、loss/metric/metadata 以及实验结论资格。
## Requirements
### Requirement: Physics supervision adapter contract
系统 SHALL 为 MMW physics-informed baseline 提供窄 adapter，将现有 dataset sample 中的 current full CSI、可选受限 CSI 输入、beamspace power label、beam power、path payload、path descriptor 和 metadata 规范化为 leakage-safe batch 字段与 `physics_targets`。adapter MUST 支持字段映射、缺失字段 mask、不可用原因和首批 shape summary，并且 MUST 不读取未启用模态的文件。

#### Scenario: 规范化可用物理监督
- **WHEN** MMW sample 包含 `csi`、`beamspace_power_label` 和 path 等价字段
- **THEN** adapter MUST 返回 `physics_targets.csi_target`、`physics_targets.beamspace_power`、`physics_targets.path_params` 和对应 valid mask
- **AND** `physics_targets.metadata` MUST 记录 scene、condition、town、sample id、field mapping 和监督来源
- **AND** 若配置启用受限 CSI 输入，adapter MUST 额外返回 `csi_input`，否则模型输入 batch MUST 不包含当前完整 CSI

#### Scenario: 缺失物理字段不使普通 batch 崩溃
- **WHEN** MMW sample 缺少 path 参数或 beam power 字段但当前配置未声明 required
- **THEN** adapter MUST 返回对应 valid mask 为 false
- **AND** 后续 physics loss MUST 跳过对应分量并记录 unavailable reason
- **AND** beam CE 训练 MUST 继续使用已有 `target_beam`

#### Scenario: required 物理字段缺失早失败
- **WHEN** 配置声明 `physics.required_fields` 包含 `csi` 或 `path_params` 且 sample 无法提供该字段
- **THEN** dataset 或 adapter MUST 抛出包含 sample id、scene、字段名和可用 keys 的错误

### Requirement: CSI and path tensor format
系统 MUST 沿用现有 CSI real/imag 末维格式。dataset 边界的 `csi_target` 和可选 `csi_input` MUST 为 finite `float32` 张量 `[T, Nsc, Nant, 2]` 或 batch 后 `[B, T, Nsc, Nant, 2]`；physics module 内部使用 PyTorch complex tensor 时，输出给 loss/diagnostics 的格式 MUST 有明确 shape metadata。path complex gain MUST 规范化为 `gain_real`、`gain_imag` 和 `path_mask`。

#### Scenario: CSI real-imag 格式进入 physics loss
- **WHEN** batch 中存在 `physics_targets.csi_target` 且 shape 为 `[B, T, Nsc, Nant, 2]`
- **THEN** physics loss MUST 将其转换为 complex tensor 时保持 autograd 兼容
- **AND** CSI NMSE 计算 MUST 使用相同 subcarrier 和 antenna 对齐口径

### Requirement: Leakage-safe CSI input contract
系统 MUST 区分三类 CSI：`csi_full_current` 表示当前时刻完整 CSI，只能默认作为 `csi_target` 监督；`csi_observed` 表示可作为模型输入的受限 CSI；`csi_history` 表示不包含当前 `H_t` 的历史 CSI 序列。默认配置 MUST 为 `data.use_csi_input=false`、`data.csi_input_mode=none`、`data.allow_oracle_full_csi_input=false`。

#### Scenario: dataset 输出字段区分输入与监督
- **WHEN** MMW sample 包含当前完整 CSI
- **THEN** dataset MUST 暴露 `csi_target` 作为当前完整 CSI 监督
- **AND** dataset MUST 暴露 `beam_label`、可选 `beam_power` 和结构化 `path_params.aod/aoa/delay/gain_real/gain_imag/path_mask`
- **AND** dataset MUST 仅在 `use_csi_input=true` 且 `csi_input_mode` 非 `none` 时暴露 `csi_input`

#### Scenario: 受限 CSI 输入不包含当前完整 CSI
- **WHEN** `data.csi_input_mode` 为 `history`、`partial`、`noisy` 或 `compressed`
- **THEN** `csi_input` MUST 由历史 CSI、部分子载波/天线 CSI、噪声 CSI 或低维压缩 CSI 生成
- **AND** `history` 模式 MUST 只包含 `H_{t-k}...H_{t-1}`，MUST 不包含当前完整 `H_t`
- **AND** 当前完整 CSI MUST 仍只作为 `csi_target` 进入 reconstruction loss、beam label 生成或 normalized beamforming gain 计算

#### Scenario: oracle full CSI 输入需要显式授权和 warning
- **WHEN** `data.csi_input_mode=oracle_full`
- **THEN** 系统 MUST 要求 `data.allow_oracle_full_csi_input=true`
- **AND** 未授权时 MUST 抛出配置或 dataset 构建错误
- **AND** 授权时 MUST 打印 `WARNING: Current full CSI is used as model input. This setting is only for oracle upper-bound baseline and may cause label leakage.`
- **AND** run metadata MUST 标记该 run 为 oracle upper-bound baseline

#### Scenario: training forward 不消费 csi_target
- **WHEN** 训练构建模型输入
- **THEN** `model_inputs` MUST 只包含 `rgb`、`depth`、`lidar`、`radar`、`imu` 和 `csi_input`
- **AND** 系统 MUST 不把 `csi_target` 传入模型 forward
- **AND** `csi_target` MUST 只传给 loss/metric，例如 CSI reconstruction、beam label 或 normalized beamforming gain

#### Scenario: path 参数标准字段
- **WHEN** path payload 使用 `AoD`、`departure_angle`、`tau`、`complex_gain` 或其它等价字段名
- **THEN** adapter MUST 根据 field map 生成标准字段 `aod`、`aoa`、`delay`、`gain_real`、`gain_imag` 和 `path_mask`
- **AND** metadata MUST 记录原始 key 到标准 key 的映射

### Requirement: Differentiable ULA channel synthesizer
系统 SHALL 提供 PyTorch 可微的 ULA array response 和 channel synthesizer。synthesizer MUST 接收 batch path 参数、subcarrier grid、antenna 数、carrier frequency、antenna spacing 和 path mask，输出 complex `h_hat`，且所有可训练预测路径到 `h_hat` 的计算 MUST 保持 autograd。

#### Scenario: ULA array response 可反传
- **WHEN** 输入 `aod_hat`、`aoa_hat`、`delay_hat`、`gain_real_hat` 和 `gain_imag_hat` 均 requires_grad
- **THEN** channel synthesizer MUST 生成 finite complex `h_hat`
- **AND** `h_hat.abs().mean().backward()` MUST 给路径参数产生 finite gradient

#### Scenario: path mask 屏蔽无效路径
- **WHEN** `path_mask` 将某些路径标记为无效
- **THEN** synthesizer MUST 不让无效路径贡献信道能量
- **AND** 输出 shape MUST 与配置的 subcarrier 和 antenna 维度一致

### Requirement: PINN multimodal beam model output contract
系统 SHALL 提供 `pinn_multimodal_beam` 模型注册名。模型 MUST 复用现有 enabled modalities 输入，输出 dict MUST 至少包含 `logits`，并通过 diagnostics 暴露 `direct_logits`、`physics_logits`、`h_hat`、`path_hat`、`latent` 和物理 shape metadata。`logits` MUST 等于配置选择的 direct、physics 或 hybrid beam logits，并能被 `adapt_model_output` 消费。

#### Scenario: hybrid logits 兼容 ModelOutput
- **WHEN** 配置启用 `use_direct_head=true`、`use_physics_head=true` 且 `physics_beta=0.5`
- **THEN** 模型 forward MUST 返回包含 `logits` 的 dict
- **AND** `adapt_model_output` MUST 将 `logits` 作为主 beam logits
- **AND** diagnostics MUST 包含 direct 与 physics 分量

#### Scenario: 可选模态输入
- **WHEN** 配置只启用 `csi` 或只启用 `image`
- **THEN** 模型 MUST 只要求对应 batch 输入
- **AND** 未启用模态缺失 MUST 不导致 forward 失败

### Requirement: Physics-informed loss bundle
系统 SHALL 提供可配置 physics-informed loss bundle。总 loss MUST 支持 beam CE、beam power distribution loss、CSI reconstruction loss、path parameter loss、array consistency loss 和可选 alignment loss。每个 physics 分量 MUST 有独立权重、valid mask、可用样本数和 diagnostic scalar；字段缺失时该分量 MUST 返回零贡献而不是伪造 target。

#### Scenario: 完整监督计算分量
- **WHEN** batch 和模型输出同时包含 beam label、`csi_target`、path params 和 beamspace power
- **THEN** loss bundle MUST 返回 `loss`、`beam_loss`、`csi_loss`、`path_loss`、`array_loss`、`beam_power_loss` 和可用样本数 diagnostics
- **AND** `loss.backward()` MUST 对模型可训练参数产生 finite gradient

#### Scenario: 缺失 CSI 跳过 reconstruction
- **WHEN** batch 不包含有效 CSI target
- **THEN** `csi_loss` MUST 为零贡献
- **AND** diagnostics MUST 标记 `csi_available_count=0`
- **AND** 总 loss MUST 继续包含 beam CE 和其它可用分量

#### Scenario: path supervision first version ordering
- **WHEN** path target 包含多个路径且未启用 matching 扩展
- **THEN** path loss MUST 按 gain magnitude 排序后监督
- **AND** metadata MUST 记录 `path_matching=sort_by_gain_magnitude`

### Requirement: Physics metrics and grouped reports
系统 SHALL 为该 baseline 提供 physics-aware evaluation metrics，包括 Top-1/Top-3/Top-5、normalized beamforming gain、CSI NMSE、path parameter MAE/Gain NMSE 和按 condition/town/scene 分组统计。指标 MUST 复用现有评估产物边界，不要求真实数据写入源码。

#### Scenario: beam power 可用时计算 normalized gain
- **WHEN** evaluation batch 包含 beam power 或 beamspace power distribution
- **THEN** metrics MUST 计算 predicted beam 的 normalized beamforming gain
- **AND** report MUST 记录 beam power source 和有效样本数

#### Scenario: group metadata 可用时输出分组指标
- **WHEN** batch metadata 包含 condition、town 或 scene
- **THEN** evaluation summary MUST 能按这些字段聚合 Top-K、normalized gain 和 physics metrics
- **AND** 缺少分组字段的样本 MUST 归入明确的 unknown group

### Requirement: Physics-aware experiment metadata
系统 SHALL 在 final config、run metadata 或 evaluation summary 中记录该 baseline 的物理配置和敏感监督使用情况。metadata MUST 包含 enabled modalities、used_csi_as_input、used_path_label_for_training、used_beam_power_for_training、used_target_physical_oracle、main_conclusion_eligible、codebook_source、array_type、num_subcarriers、num_antennas、num_paths 和 loss weights。

#### Scenario: target-side oracle 标记不可进入主结论
- **WHEN** target adaptation 或 target_test 训练阶段使用 target-side CSI、path label 或 beam power 反传
- **THEN** run metadata MUST 设置 `main_conclusion_eligible=false`
- **AND** eligibility reason MUST 指出具体使用的 target physical oracle

#### Scenario: source-only 物理监督可审计
- **WHEN** 物理监督仅用于 source split pretraining 或 supervised training
- **THEN** metadata MUST 记录 source supervision fields
- **AND** summary MUST 保留是否可与 sensor-assisted 主结论比较的机器可读字段

### Requirement: Physics configs and smoke validation
系统 SHALL 提供 physics-informed MMW debug、canonical 和 ablation 配置。每个配置 MUST 使用现有 `kd-sensing-train` 或包内 CLI 入口，MUST 不新增根目录训练脚本。实现 MUST 提供 synthetic smoke 覆盖 forward、loss、backward 和 shape summary。

#### Scenario: debug 配置跑通 synthetic batch
- **WHEN** 开发者运行 physics-informed debug focused test
- **THEN** 测试 MUST 构造 synthetic batch 并完成 forward、loss、backward
- **AND** 测试 MUST 不读取真实 `dataset/`

#### Scenario: ablation 配置关闭对应分量
- **WHEN** 用户加载 no-physics、no-CSI-reconstruction、no-path-loss 或 no-physics-head 配置
- **THEN** final config MUST 反映对应 loss weight 或 model branch 被关闭
- **AND** 关闭分量 MUST 不再贡献 loss 或 logits

### Requirement: Sparse pilot CSI input mode
系统 MUST 为 physics-informed MMW baseline 提供 `csi_input_mode=sparse_pilot`，将当前 clean CSI target 转换为带观测 mask 的 sparse pilot observation。该模式 MUST 不把未观测 CSI 值传给模型，MUST 保留完整 `csi_target` 仅用于 reconstruction supervision、beam gain 诊断或 oracle upper-bound 对照。

#### Scenario: structured sparse pilot observation
- **WHEN** 配置设置 `use_csi_input=true`、`csi_input_mode=sparse_pilot`、`pilot_pattern=grid`
- **THEN** adapter MUST 返回 `csi_input`，其 shape 与 `csi_target` 相同
- **AND** 未观测 subcarrier/antenna 位置 MUST 为 0
- **AND** adapter MUST 返回 `csi_observation_mask`，标记观测到的 pilot 位置
- **AND** metadata MUST 记录 pattern、subcarrier stride、antenna stride 和 observed fraction

#### Scenario: sparse pilot 不替代完整监督
- **WHEN** sparse pilot 输入启用
- **THEN** 模型 forward MUST 只消费 `csi_input`
- **AND** CSI reconstruction loss MUST 继续使用完整 clean `csi_target`
- **AND** run metadata MUST 将该输入标记为受限 `csi_observed`，而不是 oracle full CSI

#### Scenario: partial CSI 只作为 ablation
- **WHEN** 文档或配置描述 `partial` CSI 输入
- **THEN** 系统 SHOULD 将其标记为 debug/ablation proxy
- **AND** sparse pilot SHOULD 作为 physics-informed MMW baseline 的推荐受限 CSI 输入配置

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

### Requirement: Task-aligned physics conclusion boundary
Physics-informed MMW reports MUST distinguish multimodal CSI gains from physics-regularization gains. Results SHOULD state that sparse CSI provides the main multimodal improvement when that is what the metrics show, and MUST NOT claim large PINN gains when physics regularization only gives modest Top-1 improvement or reduces ADBA.

#### Scenario: 当前实验结论记录
- **WHEN** image+sparse CSI no-physics improves over image-only, and task-aligned PINN further improves exact Top-1 but slightly lowers ADBA
- **THEN** reports SHOULD describe CSI as the main multimodal gain
- **AND** reports SHOULD describe task-aligned physics as a modest exact-beam improvement
- **AND** reports MUST identify raw CSI reconstruction as negative transfer when full reconstruction training collapses beam metrics
