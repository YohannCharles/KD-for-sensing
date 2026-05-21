## ADDED Requirements

### Requirement: CSI information-preserving hardening
`pilot_dual_view_csi` encoder MUST support an optional `csi_hardening` configuration that applies information-preserving transforms to normalized complex CSI after train RMS normalization and before pilot-based channel estimation. The default behavior MUST remain disabled and equivalent to the current clean/noisy CSI encoder behavior.

#### Scenario: 默认关闭保持兼容
- **WHEN** encoder 配置未包含 `csi_hardening` 或设置 `csi_hardening.enabled: false`
- **THEN** encoder MUST 不应用 common phase、subcarrier phase slope、antenna calibration 或 antenna permutation
- **AND** 在其它配置相同的情况下输出 shape 和 auxiliary keys MUST 与当前 `pilot_dual_view_csi` 兼容

#### Scenario: hardening 位置在 RMS 之后 estimator 之前
- **WHEN** encoder 同时配置 `train_rms`、`csi_hardening.enabled: true` 和 pilot estimation noise
- **THEN** 系统 MUST 先将输入 CSI 除以训练集 RMS
- **AND** 系统 MUST 对归一化后的 complex CSI 应用 hardening
- **AND** 系统 MUST 再将 hardening 后的 CSI 传入 `PilotCSIChannelEstimator`

#### Scenario: hardening 输出契约
- **WHEN** 输入 CSI shape 为 `[B,T,Nsc,Nant,2]` 或 complex `[B,T,Nsc,Nant]`
- **THEN** hardening 后的 complex CSI MUST 保持 `[B,T,Nsc,Nant]`
- **AND** hardening 后的 tensor MUST 保持 finite
- **AND** encoder 最终输出 MUST 仍为 `[B,T,output_dim]`

#### Scenario: 支持信息保留型算子
- **WHEN** `csi_hardening.enabled: true`
- **THEN** encoder MUST 支持 common phase rotation、subcarrier phase slope、antenna calibration complex gain 和 fixed antenna permutation
- **AND** 每个算子 MUST 可单独启用或关闭

#### Scenario: fixed antenna transforms 可复现
- **WHEN** antenna calibration 或 antenna permutation 配置为 `fixed_by_seed`
- **THEN** 相同 seed、Nant 和配置 MUST 生成相同的 calibration 参数或 permutation
- **AND** repeated forward 在 eval 模式下 MUST 使用相同 fixed transform

#### Scenario: dataset 级 alias 归一化到 encoder
- **WHEN** 用户在 `data.dataset.csi_hardening` 中配置 hardening 且 CSI encoder 未显式配置 `csi_hardening`
- **THEN** 配置加载或训练准备阶段 MUST 将该配置复制到 teacher/student 的 `encoders.csi.csi_hardening`
- **AND** 若 encoder 内已经显式配置 `csi_hardening`，encoder 配置 MUST 优先

### Requirement: CSI encoder architecture hardening controls
`pilot_dual_view_csi` encoder MUST expose architecture controls for hard-to-learn CSI experiments while preserving the `[B,T,D]` output contract. Controls MUST include `use_internal_gru`, `view_gate_warmup_epochs`, `delay_view_warmup_epochs`, `view_fusion: freq_only`, and tokenizer capacity options.

#### Scenario: 禁用内部 GRU
- **WHEN** encoder 配置 `use_internal_gru: false`
- **THEN** encoder MUST skip its internal temporal GRU
- **AND** encoder MUST return fused per-timestep features with shape `[B,T,output_dim]`

#### Scenario: 默认使用内部 GRU
- **WHEN** encoder 未配置 `use_internal_gru`
- **THEN** encoder MUST keep the current internal GRU behavior
- **AND** existing CSI configs MUST remain valid

#### Scenario: view gate warmup
- **WHEN** encoder 配置 `view_gate_warmup_epochs: 30` 和 `view_gate_warmup_mode: mean`
- **THEN** epoch 小于 30 时 encoder MUST 使用 frequency 与 delay features 的均值融合
- **AND** epoch 达到或超过 30 时 encoder MUST 使用配置的正常 view fusion

#### Scenario: delay view warmup
- **WHEN** encoder 配置 `delay_view_warmup_epochs: 30` 和 `delay_view_warmup_mode: freq_only`
- **THEN** epoch 小于 30 时 encoder MUST 只使用 frequency view features
- **AND** epoch 达到或超过 30 时 encoder MUST 恢复 frequency + delay dual-view 处理

#### Scenario: frequency-only fusion
- **WHEN** encoder 配置 `view_fusion: freq_only`
- **THEN** encoder MUST use only frequency view features
- **AND** encoder MUST not require delay view features for the forward result

#### Scenario: tokenizer 容量配置
- **WHEN** encoder 配置 `tokenizer.hidden_channels`、`tokenizer.dropout` 或 `tokenizer.use_second_conv`
- **THEN** frequency tokenizer 与 delay tokenizer MUST 使用这些容量参数
- **AND** 未配置这些字段时 MUST 保持当前默认 tokenizer 结构

### Requirement: CSI hardening diagnostics
CSI encoder and training logs MUST expose diagnostics sufficient to analyze hardening and view fusion behavior. Diagnostics MUST be available through auxiliary output or model-side last auxiliary state and MUST be aggregated into epoch logs when present.

#### Scenario: auxiliary 输出包含 hardening 状态
- **WHEN** 调用方请求 `return_aux: true` 且 `csi_hardening.enabled: true`
- **THEN** auxiliary 输出 MUST 包含 hardening enabled 标记
- **AND** auxiliary 输出 MUST 包含至少 input power 或 phase statistic 中的一项

#### Scenario: view gate warmup diagnostics
- **WHEN** encoder 在 view gate warmup 期使用 mean fusion 且调用方请求 auxiliary 输出
- **THEN** auxiliary 输出 MUST 包含等价的 frequency/delay gate 权重
- **AND** 该权重 MUST 能表达 warmup 期两个 view 的均值融合

#### Scenario: trainer 设置 epoch 状态
- **WHEN** 训练循环开始一个 epoch
- **THEN** trainer MUST 对 student model 递归调用 epoch setter 或等价机制
- **AND** 包含 `pilot_dual_view_csi` 的模型 MUST 能基于当前 epoch 应用 warmup 行为
