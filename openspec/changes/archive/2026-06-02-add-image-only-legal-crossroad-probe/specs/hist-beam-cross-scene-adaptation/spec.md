## ADDED Requirements

### Requirement: HiST-Beam image-only variant 输出契约
HiST-Beam MUST 支持 image-only legal probe variant 或等价配置路径。该路径 MUST 复用现有 image encoder 和 projection，默认以 `identity` fusion 生成 fused image feature，并输出兼容现有 evaluator 的 logits 和 feature 字段。

#### Scenario: 构建 image-only v8/v9 probe variant
- **WHEN** 配置声明 `hist_beam.variant: image_only_v8_v9_probe` 或等价 image-only HiST-Beam probe 配置
- **THEN** 模型 MUST 只构建并消费 image 输入分支
- **AND** 模型 MUST NOT 在 forward 中访问 GPS、LiDAR、radar、mmWave、CSI、channel、path 或 beam_power
- **AND** 默认 `hist_beam.image_only.fusion_mode` MUST 为 `identity`

#### Scenario: image-only forward 输出 evaluator 兼容字段
- **WHEN** image-only model forward 完成
- **THEN** 输出 dict MUST 包含 `logits`、`logits_final` 和 `features`
- **AND** 当 target head 可用时输出 MUST 包含 `target_logits`
- **AND** 当 source head 可用时输出 MUST 包含 `source_logits`
- **AND** source-only 模式下缺失或为空的 `target_logits` MUST NOT 导致 evaluator 报错

### Requirement: Image source-only baseline
HiST-Beam quick validation MUST 支持 `run_mode: image_source_only`。该模式 MUST 使用 image-only source training 和 target_test evaluation，不执行 target adaptation，并输出标准 beam 分类指标与 collapse diagnostics。

#### Scenario: I0 source-only target eval
- **WHEN** 用户运行 I0 `image_source_only`
- **THEN** source training MUST 只使用 source image 和 beam label
- **AND** target_test evaluation MUST 只使用 target_test image 和 beam label 计算指标
- **AND** run MUST 输出 Top1、Top3、Top5、Within-1、Within-2、Within-3、MAE、prediction histogram 和 unique predicted beam 统计

### Requirement: Image-only A2 target linear probe
HiST-Beam target adaptation MUST 支持 `probe_mode: image_target_linear_probe`。该模式 MUST 从 image-only source checkpoint 初始化，冻结 image encoder、projection、optional temporal/fusion backbone 和 source head，只训练 `target_linear_head`。

#### Scenario: I1 冻结 backbone 只训练 target linear head
- **WHEN** 用户运行 I1 `image_target_linear_probe`
- **THEN** target adaptation MUST 冻结 image backbone、image projection、fusion/temporal backbone 和 source head
- **AND** optimizer MUST 只包含 `target_linear_head` 参数
- **AND** final logits MUST 等于 `target_linear_head(h_image)`

#### Scenario: I1 记录可训练参数
- **WHEN** I1 target adaptation 启动
- **THEN** 日志 MUST 输出 `[image-only A2] trainable parameter names`
- **AND** 日志或 metrics MUST 输出 `[image-only A2] trainable ratio`

### Requirement: Image-only V8 target prior head
HiST-Beam target adaptation MUST 支持 `probe_mode: image_v8_target_prior_head`。该模式 MUST 冻结 image backbone，训练 target adapter、target head、target prior bias、可学习 beta 和配置允许的 norm affine 参数；target prior MUST 只由 target support beam labels 初始化。

#### Scenario: I2 prior 只由 support labels 初始化
- **WHEN** 用户运行 I2 `image_v8_target_prior_head`
- **THEN** `target_prior_bias` MUST 由 target support beam labels 和 Gaussian smoothing 初始化
- **AND** target test labels MUST NOT 参与 prior 初始化、beta 调整、early stopping 或 target adaptation loss
- **AND** 日志 MUST 记录用于初始化 prior 的 support labels

#### Scenario: I2 final logits 不混入 source logits
- **WHEN** I2 model 计算 final logits
- **THEN** `final_logits` MUST 等于 `target_logits + beta * target_prior_bias`
- **AND** `hist_beam.v8.use_source_logits_in_final` MUST 默认为 false
- **AND** beta MUST 被 `beta_prior_max` cap，或在固定 beta 时将固定值写入日志

#### Scenario: I2 soft label 与 adapter 配置可见
- **WHEN** I2 target adaptation 启动
- **THEN** resolved config MUST 记录 `prior_sigma`、`prior_eps`、`beta_prior_init`、`beta_prior_max`、`adapter_dim`、`adapter_dropout`、`use_soft_beam_label` 和 `soft_label_sigma`
- **AND** trainable parameter metadata MUST 反映实际参与优化的 target adapter、target head、prior/beta 和允许 norm affine 参数

### Requirement: Image-only V9 sector prototype
HiST-Beam target adaptation MUST 支持 `probe_mode: image_v9_sector_proto`。该模式 MUST 从 target support image feature 按 sector 建 prototype，默认不启用 beam-level prototype，并将 sector prototype logits 映射回 beam logits 参与 final logits。

#### Scenario: I3 构建 sector prototype
- **WHEN** 用户运行 I3 `image_v9_sector_proto`
- **THEN** 系统 MUST 用 target support image feature 构建 prototype
- **AND** `sector_label` MUST 按 `beam_label // sector_size` 计算
- **AND** prototype MUST 为同一 sector 中 normalized support features 的均值
- **AND** 默认 `sector_size` MUST 为 2 或 3
- **AND** `hist_beam.v9.use_beam_proto` MUST 默认为 false

#### Scenario: I3 sector logits 映射回 beam logits
- **WHEN** I3 对 target query/test feature 计算 prototype score
- **THEN** 系统 MUST 使用 cosine similarity 除以 `proto_temperature` 得到 sector score
- **AND** 每个 beam 的 proto logit MUST 使用其所属 sector 的 score
- **AND** 无 prototype 的 sector MUST 使用 0 或明确配置的小值作为 proto logit

#### Scenario: I3 final logits 与日志
- **WHEN** I3 model 计算 final logits
- **THEN** `final_logits` MUST 等于 `target_logits + beta * target_prior_bias + eta * sector_proto_logits`
- **AND** 日志 MUST 输出 `[v9-sector] support labels`
- **AND** 日志 MUST 输出 `[v9-sector] support sectors`
- **AND** 日志 MUST 输出 `[v9-sector] prototype sectors`
- **AND** 日志 MUST 输出 `[v9-sector] top predicted beams before proto`
- **AND** 日志 MUST 输出 `[v9-sector] top predicted beams after proto`

### Requirement: Image-only adaptation 设备与 dtype 稳定
HiST-Beam image-only legal probe MUST 保持 tensor device 和 dtype 兼容 bf16/fp16 混合精度。feature cache 若保存低精度 feature，metadata MUST 明确 dtype；默认保存前 MUST 转为 fp32。

#### Scenario: feature cache dtype 可审计
- **WHEN** image feature cache 写出
- **THEN** cache metadata MUST 记录 feature dtype
- **AND** 若运行使用 bf16/fp16，保存到磁盘的 feature MUST 为 fp32 或 metadata MUST 明确记录低精度 dtype 与读取转换策略

#### Scenario: loss backward smoke test
- **WHEN** image-only target adaptation smoke test 执行
- **THEN** loss backward MUST 在当前 device 和 dtype 设置下成功
- **AND** smoke test 命令 MUST 使用 `conda run -n kd_mm_beam`
