# csi-modality-model Specification

## Purpose
TBD - created by archiving change add-pilot-dual-view-csi-modality. Update Purpose after archive.
## Requirements
### Requirement: Pilot-based CSI channel estimator
系统 MUST 提供 pilot-based CSI 估计模块，将 clean CSI `h` 转换为 noisy channel estimate `h_hat = h + e`。在物理参数模式下，估计噪声方差 MUST 满足 `sigma_e2 = noise_var / (pilot_power * pilot_len)`；在 estimation SNR 模式下，噪声方差 MUST 由 clean CSI 信号功率和 `snr_db` 决定。

#### Scenario: 物理参数模式
- **WHEN** estimator 配置 `mode: physical`、`pilot_len: 16`、`pilot_power: 1.0` 和 `noise_var: 0.01`
- **THEN** estimator MUST 使用 `0.01 / 16` 作为 complex estimation noise 方差
- **AND** real 和 imag 噪声分量方差 MUST 分别为 complex 方差的一半

#### Scenario: estimation SNR 训练采样
- **WHEN** estimator 处于 training 模式且配置 `train_snr_min_db` 与 `train_snr_max_db`
- **THEN** estimator MUST 为 batch 采样该区间内的 SNR
- **AND** 输出 auxiliary diagnostics MUST 包含本次使用的 `snr_db` 或等价张量

#### Scenario: 无噪声配置返回 clean estimate
- **WHEN** estimator 未配置 `noise_var`、`est_snr_db` 或训练 SNR 区间
- **THEN** estimator MUST 返回归一化后的 clean CSI
- **AND** auxiliary diagnostics MUST 表明 `sigma_e2` 为 0 或未启用噪声

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
项目 MUST 提供 CSI-only 基线配置和至少一个包含 CSI 的 fusion 示例配置。配置 MUST 暴露 clean/noisy、SNR、pilot length、pilot power、delay taps 和 view fusion 参数。

#### Scenario: CSI-only no-KD 配置可加载
- **WHEN** 用户加载 CSI-only no-KD 配置
- **THEN** 配置 MUST 设置 `experiment.task: csi`
- **AND** 配置 MUST 使用 `modular_sequence` 和 `pilot_dual_view_csi` encoder

#### Scenario: SNR 消融配置
- **WHEN** 用户配置评估 SNR 为 `0`、`5`、`10`、`20` 或 `30` dB
- **THEN** 系统 MUST 能将固定 `snr_db` 传入 CSI estimator
- **AND** 运行 metadata 或 diagnostics MUST 记录评估使用的 SNR

