# gps-conditioned-jepa-pretraining Specification

## Purpose
定义 GPS-Rel-Polar 条件化 JEPA 预训练能力的当前契约，覆盖主模型注册、GPS 条件化 latent prediction、mask 采样、EMA target encoder、训练产物以及下游 supervised fusion 复用边界，确保该自监督入口不依赖 beam target、蒸馏 teacher 或历史 checkpoint 脚手架。
## Requirements
### Requirement: GPS 条件化 JEPA 主模型
系统 MUST 提供可通过 `MODELS` 注册表构建的 GPS-conditioned JEPA 主模型。该模型 MUST 作为 `model.primary` 使用，内部包含 context visual encoder、EMA target visual encoder、GPS conditioner、latent predictor 和 mask sampler。该模型 MUST 接收 image 输入 `[B, T, 3, H, W]` 与 GPS-Rel-Polar 输入 `[B, T, 3]`，并 MUST 不要求 `target_beam`、beam logits、distiller 或外部 frozen teacher checkpoint。

#### Scenario: 构建 JEPA 主模型
- **WHEN** 用户配置 `model.primary.type: gps_conditioned_jepa`
- **THEN** 系统 MUST 通过 `MODELS.build()` 返回 JEPA 主模型实例
- **AND** 构建参数 MUST 支持 visual encoder、GPS conditioner、predictor、mask sampler、latent dimension、EMA decay 和 image/GPS input profile 配置

#### Scenario: JEPA forward 输入契约
- **WHEN** JEPA 主模型接收 `image_batch` 形状为 `[B, T, 3, H, W]` 且 `gps_batch` 形状为 `[B, T, 3]`
- **THEN** 模型 MUST 输出 context/target latent prediction payload
- **AND** 输出 MUST 至少包含 `predicted_target_latent`、`target_latent`、`context_mask`、`target_mask`、`loss_mask` 和 scalar diagnostics
- **AND** `predicted_target_latent` 与 `target_latent` MUST 具有相同形状 `[B, T, N_tgt, D]`

#### Scenario: 缺失 GPS 输入时报错
- **WHEN** JEPA 主模型 forward 未收到 `gps_batch`
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出 GPS-conditioned JEPA 需要 GPS-Rel-Polar 输入

### Requirement: JEPA target encoder EMA 契约
JEPA target encoder MUST 与 context encoder 同构，并 MUST 初始化为 context encoder 的参数副本。target encoder 参数 MUST 不参与梯度更新，且 MUST 在 optimizer step 后通过 EMA 从 context encoder 更新。EMA 状态 MUST 能随 checkpoint 保存和恢复。

#### Scenario: target encoder 不接收梯度
- **WHEN** 一次 JEPA 训练 batch 完成 forward 和 backward
- **THEN** target encoder 参数的 `requires_grad` MUST 为 false
- **AND** target latent MUST 从 autograd graph 中 detach

#### Scenario: optimizer step 后更新 EMA
- **WHEN** optimizer 完成一次 context encoder 参数更新
- **THEN** JEPA runtime MUST 使用配置的 EMA decay 更新 target encoder 参数
- **AND** EMA 更新 MUST 发生在 optimizer step 之后

#### Scenario: resume 后保留 EMA 状态
- **WHEN** 用户从 JEPA checkpoint resume 训练
- **THEN** checkpoint MUST 恢复 context encoder、target encoder、predictor、conditioner、optimizer、scheduler 和已完成 epoch
- **AND** 继续训练时 target encoder MUST 从恢复状态继续 EMA 更新

### Requirement: GPS 条件化 latent prediction
JEPA predictor MUST 使用 GPS-Rel-Polar 特征调制 context latent，再预测 target patch/token latent。默认 conditioning MUST 支持 FiLM 调制，配置 MAY 选择 concat-MLP 等已实现 conditioner。conditioning MUST 不读取 raw latitude/longitude，也不得绕过当前 GPS feature mode 契约。

#### Scenario: FiLM conditioning 输出形状
- **WHEN** conditioner 类型为 `film` 且输入 context latent 为 `[B, T, N_ctx, D]`
- **THEN** conditioner MUST 从 GPS tensor 生成与 latent dimension 对齐的调制参数
- **AND** conditioned context latent MUST 保持 `[B, T, N_ctx, D]` 形状

#### Scenario: GPS feature 维度校验
- **WHEN** GPS tensor 最后一维不等于配置的 `gps_input_size`
- **THEN** 系统 MUST 拒绝 forward
- **AND** 错误信息 MUST 包含实际 GPS 维度和期望 GPS 维度

### Requirement: JEPA patch mask 采样
系统 MUST 提供 JEPA patch/token mask sampler。sampler MUST 支持 `random` 和 `gps_angle_biased` 两种初始模式，并 MUST 输出 context mask、target mask 和 loss mask。context 与 target token 在同一帧内 MUST 不重叠，mask 采样 MUST 可通过 run seed、epoch 和 step 复现。

#### Scenario: random mask 采样
- **WHEN** mask sampler 配置为 `mode: random`
- **THEN** sampler MUST 按配置的 context ratio 和 target ratio 采样 token
- **AND** context mask 与 target mask MUST 在同一帧内不重叠

#### Scenario: GPS angle biased mask 采样
- **WHEN** mask sampler 配置为 `mode: gps_angle_biased`
- **THEN** sampler MUST 使用 GPS relative angle 对 patch grid 构造采样权重
- **AND** diagnostics MUST 记录 mask mode、context ratio、target ratio 和有效 target token 数

#### Scenario: mask 采样可复现
- **WHEN** 使用相同 run seed、epoch、step、batch index 和输入 shape 调用 sampler
- **THEN** sampler MUST 产生相同的 context mask 和 target mask

### Requirement: JEPA latent loss 和诊断
系统 MUST 提供 JEPA latent prediction loss。loss MUST 只在 `loss_mask` 有效的 target token 上计算，默认使用 latent-space MSE，并 MUST 支持可配置的 latent normalization、Huber/SmoothL1 变体和 loss 权重。训练和验证 MUST 记录 JEPA loss 与关键 mask/EMA diagnostics。

#### Scenario: latent MSE loss
- **WHEN** predicted target latent 与 target latent 形状一致且 loss mask 有有效 token
- **THEN** JEPA loss MUST 在有效 token 上计算平均 latent MSE
- **AND** 返回 diagnostics MUST 包含 `loss/jepa` 或等价正式字段

#### Scenario: 无有效 target token
- **WHEN** loss mask 中没有有效 target token
- **THEN** 系统 MUST 抛出清晰错误或跳过该 batch 并记录诊断
- **AND** 系统 MUST 不静默产生 NaN loss

### Requirement: JEPA 预训练产物
JEPA 预训练运行 MUST 保存完整训练产物，并 MUST 显式记录可复用 context encoder 信息。运行产物 MUST 保持在 `outputs/`、`logs/` 或配置指定的运行目录内，不得要求提交 checkpoint、cache 或 TensorBoard 文件。

#### Scenario: 保存 JEPA checkpoint
- **WHEN** JEPA 预训练完成至少一个 epoch
- **THEN** 运行目录 MUST 包含可恢复训练的 checkpoint
- **AND** checkpoint MUST 包含 context encoder、target encoder、GPS conditioner、predictor、optimizer、scheduler 和 objective metadata

#### Scenario: 记录 context encoder 可复用信息
- **WHEN** JEPA 预训练写出 `final_config.yaml` 或运行 metadata
- **THEN** metadata MUST 记录 visual token encoder 类型、latent dimension、mask sampler 配置、GPS conditioner 配置和 context encoder state dict key
- **AND** metadata MUST 指明该 checkpoint 来源为 `gps_conditioned_jepa`

### Requirement: JEPA context encoder 下游复用
系统 MUST 提供 supervised fusion 可配置使用的 JEPA context image encoder 初始化入口。该入口 MUST 从 JEPA checkpoint 中抽取 `context_encoder` 权重，MUST 同时兼容仅包含 state dict 的 objective checkpoint 和包含 `state_dict` 的恢复 checkpoint。默认模式 MUST 输出现有 fusion representation core 可消费的帧级 image 特征 `[B,T,D]`，并保持 patch-token mean pooling 兼容；显式配置 downstream pooler/adapter 时，MUST 通过可配置 pooler/adapter 复用 context encoder patch tokens，同时 MUST 不要求重训 JEPA Stage 1、target encoder EMA 或 latent prediction loss。

#### Scenario: 从 JEPA best checkpoint 初始化 supervised image encoder
- **WHEN** supervised fusion 配置将 image encoder 设置为 `jepa_context_image` 且 checkpoint 指向 JEPA `best.pth`
- **THEN** 系统 MUST 加载 checkpoint 中的 `context_encoder.*` 权重
- **AND** image encoder forward MUST 将 `[B, T, 3, H, W]` 输入转换为 `[B, T, D]` 特征
- **AND** 未显式声明 downstream pooler 时 MUST 使用 mean pooling 以保持既有配置兼容

#### Scenario: 从 JEPA last checkpoint 初始化 supervised image encoder
- **WHEN** supervised fusion 配置将 image encoder 设置为 `jepa_context_image` 且 checkpoint 指向 JEPA `last.pth`
- **THEN** 系统 MUST 从 checkpoint payload 的 `state_dict` 字段抽取 `context_encoder.*` 权重
- **AND** 输出特征维度 MUST 与 fusion 配置的 image encoder `output_dim` 一致

#### Scenario: 通过 pooler 复用 JEPA patch tokens
- **WHEN** supervised fusion 配置将 image encoder 设置为 `jepa_context_image` 并显式声明 downstream pooler
- **THEN** 系统 MUST 继续只加载 checkpoint 中的 `context_encoder.*` 权重作为 JEPA 视觉 backbone
- **AND** 系统 MUST 将 context encoder 产生的 patch tokens `[B,T,N,D]` 传给配置的 pooler
- **AND** pooler 默认输出 MUST 保持 `[B,T,D]`
- **AND** 系统 MUST NOT 要求 JEPA target encoder、EMA 更新、JEPA latent loss、distiller 或外部 frozen teacher

#### Scenario: GPS-query pooler 缺少条件特征
- **WHEN** supervised fusion 配置启用需要 GPS 条件特征的 JEPA downstream pooler，但 image encoder forward 未收到 GPS 条件特征
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出 `jepa_context_image` 的该 pooler 需要 GPS condition feature

#### Scenario: fair 下游配置复用多场景 JEPA checkpoint
- **WHEN** supervised BeamBench-fair JEPA 复用配置被加载
- **THEN** random-mask JEPA 配置 MUST 指向 `outputs/deepsense6g_gps_conditioned_jepa_full_s32_s34_lowmem/checkpoints/best.pth` 或 `last.pth`
- **AND** GPS-biased JEPA 配置 MUST 指向 `outputs/deepsense6g_gps_conditioned_jepa_gps_biased_s32_s34_lowmem/checkpoints/best.pth`
- **AND** 配置 MUST NOT 默认引用 scene31-only JEPA checkpoint
- **AND** JEPA downstream 派生配置 MUST 继承与 baseline 匹配的多场景 checkpoint 口径
- **AND** BeamBench-fair downstream 配置 MUST 继承 `seq_len=1`、`num_pred=1`、GPS `paper_distance_angle`、scene paper calibration angle 和 linear DBA 口径

### Requirement: JEPA visual token encoder variants
GPS-conditioned JEPA 主模型 MUST 支持可配置 visual token encoder variants。默认 variant MUST 保持现有 patch16 `VisualPatchTokenEncoder` 行为兼容；opt-in variants MAY 包含 overlap patch tokenizer、conv stem tokenizer、local token mixing、CvT-style convolutional projection、CNN feature-map tokens 或多尺度 tokens。所有 variants MUST 输出可供 JEPA predictor 和 mask sampler 消费的 `[B,T,N,D]` tokens。

#### Scenario: 默认 patch encoder 兼容
- **WHEN** 用户未设置 `visual_encoder.type` 或设置为现有 patch encoder 等价类型
- **THEN** 系统 MUST 保持现有 patch16 非重叠 tokenization、checkpoint loading 和 forward shape 行为兼容
- **AND** 现有 GPS-conditioned JEPA 配置 MUST 无需修改即可构建和训练

#### Scenario: opt-in tokenizer 输出统一 token 契约
- **WHEN** 用户配置 overlap patch、conv stem、local token mixing、CvT-style 或 CNN-token visual encoder variant
- **THEN** encoder MUST 输出 `[B,T,N,D]` tokens 和 token/grid metadata
- **AND** metadata MUST 记录 `visual_encoder.type`、image size、effective stride、token grid、token count、positional encoding 和 max token budget

#### Scenario: token budget 超限时报错
- **WHEN** visual encoder variant 产生的 token 数超过配置的 `max_tokens` 或模型预算
- **THEN** 系统 MUST 抛出包含实际 token count、max token count、image size 和 variant type 的清晰错误
- **AND** 系统 MUST 不静默截断 tokens

### Requirement: JEPA mask sampler 适配可变 token grid
JEPA mask sampler MUST 基于 visual token encoder 提供的 token/grid metadata 采样 context 和 target tokens。sampler MUST 不硬编码 patch16、14x14 或 196 tokens，并 MUST 在 GPS angle biased mode 下记录使用的 token grid。

#### Scenario: GPS angle biased mask 使用 token metadata
- **WHEN** visual encoder variant 的 token grid 不是 14x14
- **THEN** GPS angle biased mask sampler MUST 使用该 variant 的 token/grid metadata 构造采样权重
- **AND** diagnostics MUST 记录 mask mode、token grid、context ratio、target ratio 和有效 target token 数

#### Scenario: 多尺度 token 可审计
- **WHEN** visual encoder 输出多尺度 tokens 或合并后的 token sequence
- **THEN** mask sampler diagnostics MUST 记录每个 scale 的 token count 或合并策略
- **AND** predictor target shape MUST 与被采样 target tokens 对齐

### Requirement: JEPA visual encoder checkpoint policy
GPS-conditioned JEPA 预训练和下游复用 MUST 显式记录 visual encoder checkpoint policy。policy MUST 区分 `exact_reuse`、`partial_reuse`、`pos_interpolate`、`fresh_stage1_required` 和 `supervised_only_anchor` 或等价状态。

#### Scenario: 形状不匹配不能伪装为 exact reuse
- **WHEN** visual encoder variant 的 patch embedding、position embedding 或 backbone 参数形状与 checkpoint 不匹配
- **THEN** 系统 MUST 不允许将该 run 标记为 `exact_reuse`
- **AND** metadata MUST 记录 missing keys、unexpected keys、interpolated position grid 或 fresh initialization reason

#### Scenario: 新 tokenizer 需要新 Stage 1 checkpoint
- **WHEN** tokenizer/backbone 改变导致无法复用现有 GPS-biased JEPA context encoder
- **THEN** 配置或 metadata MUST 标记 `fresh_stage1_required`
- **AND** downstream strict comparison MUST 使用该 tokenizer 对应的 Stage 1 checkpoint，而不是旧 patch16 checkpoint

