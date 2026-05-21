## MODIFIED Requirements

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

## ADDED Requirements

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
