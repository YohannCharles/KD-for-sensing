# csi-modality-model Specification

## Purpose
定义 CSI 模态模型、pilot dual-view encoder、估计/硬化配置和训练诊断契约。
## Requirements
### Requirement: Pilot-based CSI channel estimator
系统 MUST 提供 pilot-based CSI 估计模块，将 clean CSI `h` 转换为 noisy channel estimate `h_hat = h + e`。在物理参数模式下，估计噪声方差 MUST 满足 `sigma_e2 = noise_var / (pilot_power * pilot_len)`，且该 `noise_var` MUST 与 estimator 接收的 CSI 张量尺度一致。在 estimation SNR 模式下，噪声方差 MUST 由 estimator 输入 CSI 的信号功率和 `snr_db` 决定。用于 mild pilot estimation 的实验配置 MUST 使用 estimation SNR 或经过噪声/信号比诊断证明为 mild 的物理方差。

#### Scenario: 物理参数模式
- **WHEN** estimator 配置 `mode: physical`、`pilot_len: 16`、`pilot_power: 1.0` 和 `noise_var: 0.01`
- **THEN** estimator MUST 使用 `0.01 / 16` 作为 complex estimation noise 方差
- **AND** real 和 imag 噪声分量方差 MUST 分别为 complex 方差的一半
- **AND** auxiliary diagnostics MUST 记录 `sigma_e2`、`h_power_mean`、`noise_power_mean` 和 `noise_power_signal_ratio`

#### Scenario: estimation SNR 训练采样
- **WHEN** estimator 处于 training 模式且配置 `train_snr_min_db` 与 `train_snr_max_db`
- **THEN** estimator MUST 为 batch 采样该区间内的 SNR
- **AND** 输出 auxiliary diagnostics MUST 包含本次使用的 `snr_db` 或等价张量
- **AND** `noise_power_signal_ratio` MUST 与采样 SNR 对应的功率比同量级

#### Scenario: mild pilot estimation 使用相对 SNR
- **WHEN** CSI hardening matrix 的 A1 类 mild pilot estimation 配置被加载
- **THEN** 配置 MUST 使用 `mode: est_snr` 或 `mode: estimation_snr`
- **AND** 配置 MUST 提供固定 `snr_db` 或训练 SNR 采样区间
- **AND** 运行 diagnostics MUST 能证明 `noise_power_signal_ratio` 落在该 SNR 对应范围内

#### Scenario: 归一化输入上的物理噪声失真
- **WHEN** estimator 输入已经过训练集 RMS 归一化且配置使用 `mode: physical`
- **THEN** diagnostics MUST 记录实际 `noise_power_signal_ratio`
- **AND** 如果该 run 被标记为 mild pilot estimation，系统 MUST 能基于 diagnostics 将明显超出 mild 区间的噪声标记为 invalid pilot noise scale

#### Scenario: 无噪声配置返回 clean estimate
- **WHEN** estimator 未配置 `noise_var`、`est_snr_db` 或训练 SNR 区间
- **THEN** estimator MUST 返回归一化后的 clean CSI
- **AND** auxiliary diagnostics MUST 表明 `sigma_e2` 为 0 或未启用噪声
- **AND** `pilot_identity_max_abs` MUST 为 0 或浮点精度内的 0

### Requirement: Pilot dual-view CSI encoder 结构
系统 MUST 提供 `pilot_dual_view_csi` encoder，用于将 CSI 输入编码为 `[B, T, D]` 特征。该 encoder MUST 先执行训练集 RMS 归一化和 pilot-based channel estimation，再从 `h_hat` 生成 frequency view 与 delay view。

#### Scenario: real/imag 输入前向输出
- **WHEN** encoder 接收 `[B, T, Nsc, Nant, 2]` CSI 输入
- **THEN** encoder MUST 输出 `[B, T, output_dim]`
- **AND** `output_dim` MUST 等于配置的 `output_dim`、`d_model` 或 `feature_size`

#### Scenario: complex 输入前向输出
- **WHEN** encoder 接收 complex dtype 的 `[B, T, Nsc, Nant]` CSI 输入
- **THEN** encoder MUST 正确解析为复数信道
- **AND** 输出 MUST 仍为 `[B, T, output_dim]`

#### Scenario: 先估计再 IFFT
- **WHEN** encoder 启用 delay view
- **THEN** delay view MUST 从 noisy estimate `h_hat` 沿 subcarrier 维执行 IFFT
- **AND** 系统 MUST 不先对 clean CSI 做 IFFT 后再分别加噪

### Requirement: CSI 双视图 tokenizer 与融合
CSI encoder MUST 为 frequency view 和 delay view 使用独立参数的 CNN tokenizer，并 MUST 支持 `mean`、`concat` 和 `symmetric_gate` 三种 view fusion。默认 view fusion MUST 为 `symmetric_gate`。

#### Scenario: frequency view shape
- **WHEN** `h_hat` 的形状为 `[B, T, Nsc, Nant]`
- **THEN** frequency view MUST 形成 `[B, T, 2, Nant, Nsc]`

#### Scenario: delay view taps
- **WHEN** `delay_taps: 32` 且 `Nsc` 小于 32
- **THEN** delay view MUST 使用 `min(delay_taps, Nsc)` 个 tap
- **AND** 输出 view MUST 形成 `[B, T, 2, Nant, L_delay]`

#### Scenario: symmetric gate diagnostics
- **WHEN** view fusion 为 `symmetric_gate` 且调用方请求 auxiliary 输出
- **THEN** encoder MUST 返回 frequency/delay gate 或等价 diagnostics
- **AND** gate 最后一维 MUST 对应两个 view 的权重

### Requirement: CSI encoder 注册与 modular_sequence 集成
CSI encoder MUST 通过 `ENCODERS` 注册表以 `pilot_dual_view_csi` 名称构建，并 MUST 能作为 `modular_sequence` 的 `encoders.csi` 分支使用。

#### Scenario: 按注册表构建 CSI encoder
- **WHEN** 配置指定 `encoders.csi.type: pilot_dual_view_csi`
- **THEN** 系统 MUST 通过 `ENCODERS` 注册表返回 CSI encoder 实例
- **AND** encoder MUST 暴露 `output_dim`

#### Scenario: CSI-only modular sequence 模型
- **WHEN** 配置指定 `type: modular_sequence` 且 `modalities: [csi]`
- **THEN** 模型 MUST 接收 `csi_batch`
- **AND** 输出 logits MUST 具有 `[B, T, num_classes]` 形状

### Requirement: CSI 实验配置与消融参数
项目 MUST 提供 CSI-only supervised 配置和至少一个包含 CSI 的 fusion 示例配置，使用户能构建 CSI-only primary model 并运行训练或评估。该配置 MUST 不使用 no-KD 或 distillation 命名，并 MUST 暴露 clean/noisy、SNR、pilot length、pilot power、delay taps 和 view fusion 参数。

#### Scenario: CSI-only supervised 配置可加载
- **WHEN** 用户加载 CSI-only supervised 配置
- **THEN** 配置 MUST 设置 `experiment.task: csi`
- **AND** 配置 MUST 使用 `modular_sequence` 和 `pilot_dual_view_csi` encoder
- **AND** 配置 MUST 不包含 `distillation.type`
- **AND** 配置 MUST 不要求 teacher checkpoint

#### Scenario: SNR 消融配置
- **WHEN** 用户配置评估 SNR 为 `0`、`5`、`10`、`20` 或 `30` dB
- **THEN** 系统 MUST 能将固定 `snr_db` 传入 CSI estimator
- **AND** 运行 metadata 或 diagnostics MUST 记录评估使用的 SNR

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

### Requirement: CSI pilot disabled identity diagnostics
CSI pilot estimator MUST expose diagnostics that prove disabled pilot estimation leaves the normalized complex CSI unchanged. When pilot estimation is disabled, `h_hat` MUST be exactly equal to the input `h` within floating point precision and the debug output MUST include `max_abs(h_hat - h)`.

#### Scenario: pilot disabled returns identity
- **WHEN** `pilot_estimator.enabled` is `false`
- **THEN** `PilotCSIChannelEstimator` MUST return `h_hat` equal to the input normalized CSI
- **AND** debug diagnostics MUST report `max_abs(h_hat - h)` as 0 or floating point zero

#### Scenario: mild pilot SNR records noise ratio
- **WHEN** pilot estimation is enabled with training SNR sampled between 25 dB and 35 dB
- **THEN** diagnostics MUST record sampled SNR or equivalent SNR tensor
- **AND** diagnostics MUST record `noise_power/signal_power`
- **AND** the expected ratio SHOULD be approximately between 0.003 and 0.0003 before stochastic tolerance is applied

### Requirement: CSI hardening invariant diagnostics
CSI hardening diagnostics MUST verify that hardening preserves complex CSI shape, finite values and expected scale unless a configuration explicitly requests gain scaling. The diagnostics MUST be available for each enabled hardening transform.

#### Scenario: hardening preserves shape and finite values
- **WHEN** hardening is applied to normalized complex CSI with shape `[B,T,Nsc,Nant]`
- **THEN** the hardening output MUST keep shape `[B,T,Nsc,Nant]`
- **AND** diagnostics MUST report `nan_count=0`
- **AND** diagnostics MUST report zero ratio and magnitude statistics

#### Scenario: hardening scale drift warning
- **WHEN** hardening output abs_mean or abs_std changes by more than 20 percent relative to hardening input without explicit gain scaling
- **THEN** diagnostics MUST mark the batch as suspicious
- **AND** the warning MUST include before and after abs_mean and abs_std values

#### Scenario: fixed antenna transforms are not resampled per batch
- **WHEN** antenna calibration or fixed antenna permutation uses a fixed seed
- **THEN** the transform MUST remain stable across batches in the same run
- **AND** diagnostics MUST expose enough transform identity information to detect accidental per-batch resampling

### Requirement: CSI encoder path structure diagnostics
`pilot_dual_view_csi` MUST report its resolved structure at run start when model debug summary is enabled. The summary MUST include `use_internal_gru`, `view_fusion`, `delay_taps`, `d_model`, total parameters and trainable parameters.

#### Scenario: default internal GRU is visible
- **WHEN** a run builds a CSI encoder without explicitly setting `use_internal_gru`
- **THEN** the resolved summary MUST show `use_internal_gru=true`
- **AND** the encoder MUST keep its existing internal GRU path

#### Scenario: no internal GRU path remains connected
- **WHEN** a run sets `use_internal_gru=false`
- **THEN** the resolved summary MUST show the no-internal-GRU path
- **AND** the encoder output MUST still have shape `[B,T,D]`
- **AND** final CSI feature norm diagnostics MUST be nonzero for nonzero CSI input

### Requirement: CSI view fusion warmup diagnostics
CSI view fusion warmup MUST preserve nonzero feature flow and expose gate/fusion diagnostics. Warmup MUST not zero both views or produce a zero fused feature for nonzero CSI input.

#### Scenario: view gate warmup keeps feature flow
- **WHEN** `view_gate_warmup_epochs` is active and the input CSI batch is nonzero
- **THEN** diagnostics MUST report nonzero freq_feat norm
- **AND** diagnostics MUST report nonzero delay_feat norm unless delay view is separately disabled
- **AND** diagnostics MUST report nonzero fused_feat norm

#### Scenario: gate broadcast is valid
- **WHEN** `view_fusion=symmetric_gate` or warmup mean fusion is active
- **THEN** gate or equivalent fusion weights MUST broadcast over `[B,T,D]` features correctly
- **AND** diagnostics MUST not report NaN values for gate, fused feature or final CSI feature

### Requirement: CSI hardening matrix pilot isolation
CSI hardening matrix 中非 pilot-only 的 A/B/C/D 单变量配置 MUST 显式隔离 pilot estimation 噪声，避免 hardening、encoder 结构和 pilot noise 被混合解释。除明确测试 pilot estimation 的 A1 及 destructive negative control 外，CSI-only hardening/encoder 配置 MUST 显式设置 `csi_estimation.mode: none` 或等价 disabled 状态。

#### Scenario: hardening-only 配置关闭 pilot noise
- **WHEN** 系统加载 B 组 hardening-only 配置
- **THEN** 配置中的 CSI encoder MUST 显式关闭 pilot estimation noise
- **AND** 首 batch diagnostics MUST 显示 `noise_power_signal_ratio=0` 或等价无噪声状态

#### Scenario: encoder-only 配置关闭 pilot noise
- **WHEN** 系统加载 C 组 encoder-only 配置
- **THEN** 配置中的 CSI encoder MUST 显式关闭 pilot estimation noise
- **AND** C1 view gate warmup 和 C2 no internal GRU 的输入 CSI MUST 与 A0 clean baseline 的 pilot 后 CSI 等价

#### Scenario: combined hardening 配置只组合声明变量
- **WHEN** 系统加载 D 组 hardening+encoder combined 配置
- **THEN** 配置 MUST 只组合对应 hardening 和 encoder 变量
- **AND** 配置 MUST 不隐式继承 A1 的 pilot estimation noise

### Requirement: Fusion 输入准备支持 CSI
训练、验证和评估流程在 `experiment.task: fusion` 下 MUST 能根据配置的 `modalities` 准备 CSI 输入。未启用 CSI 时，batch 准备和模型 forward MUST 不要求 `csi` 或 `csi_batch`。

#### Scenario: fusion 启用 CSI 和 GPS
- **WHEN** fusion 配置的 `modalities` 为 `["gps", "csi"]`
- **THEN** batch 准备 MUST 构造 `gps_batch` 和 `csi_batch`
- **AND** batch 准备 MUST 不要求 image、radar、LiDAR 或 mmWave 字段

#### Scenario: fusion 启用全部六模态
- **WHEN** fusion 配置的 `modalities` 为 `["image", "radar", "gps", "lidar", "mmwave", "csi"]`
- **THEN** batch 准备 MUST 构造六个模态输入
- **AND** 六个输入的 batch 和 sequence 维度 MUST 对齐

### Requirement: Modular fusion 使用 CSI encoder 输出
`modular_sequence` fusion 模型 MUST 能将 CSI encoder 的 `[B, T, D]` 输出与其它模态 encoder 输出对齐，并通过既有 projector 和 representation core 处理。

#### Scenario: modular_sequence 融合 CSI 与 mmWave
- **WHEN** 配置 `modalities: ["mmwave", "csi"]`
- **THEN** 模型 MUST 分别调用 mmWave encoder 和 CSI encoder
- **AND** 两个 projected feature MUST 在 batch、time 和 `d_model` 维度上兼容

### Requirement: CSI 组件注册
项目 MUST 通过现有组件注册表注册 CSI encoder 和可选 CSI 模型入口，使用户能通过配置构建 pilot dual-view CSI encoder，并复用现有 `modular_sequence` 训练流程。

#### Scenario: 按名称构建 CSI encoder
- **WHEN** 配置中指定 `type: pilot_dual_view_csi` 及其初始化参数
- **THEN** 系统 MUST 通过 `ENCODERS` 注册表返回 CSI encoder 实例
- **AND** 构建参数 MUST 支持 `output_dim`、`d_model`、pilot estimation、dual-view、tokenizer、temporal 和 dropout 相关字段

#### Scenario: 默认组件导入包含 CSI 模块
- **WHEN** 构建流程调用默认组件导入函数后再构建 `pilot_dual_view_csi`
- **THEN** `ENCODERS` 注册表 MUST 包含 `pilot_dual_view_csi`
- **AND** 注册表轻量导入边界 MUST 与现有 registry 语义一致

### Requirement: CSI 注册错误可诊断
CSI 相关注册错误 MUST 使用现有注册表错误风格，并在未知名称、重复名称或缺失必需参数时提供清晰错误信息。

#### Scenario: 请求未知 CSI encoder
- **WHEN** 配置中引用未注册的 CSI encoder 名称
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含请求的组件名称、注册表名称和可用组件列表

#### Scenario: CSI 构建参数非法
- **WHEN** 配置中引用 `pilot_dual_view_csi` 但提供非法 `view_fusion` 或非正数 `pilot_len`
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含非法字段或原始构建错误
