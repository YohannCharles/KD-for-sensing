# csi-channel-degradation Specification

## Purpose
定义 CSI degradation profile、配置开关和质量诊断要求，使 CSI 扰动实验的输入变换、记录字段和验证结果保持可复核。
## Requirements
### Requirement: CSI degradation 配置与质量 profile
系统 MUST 提供可配置的 CSI degradation 能力。该能力 MUST 默认关闭；启用时 MUST 支持 `clean`、`medium` 和 `hard` profile，并 MUST 允许 YAML 配置覆盖 profile 中的单项参数。系统 MUST 在最终配置或运行 metadata 中记录解析后的有效 degradation 参数。

#### Scenario: 默认不退化 CSI
- **WHEN** dataset 配置未提供 `csi_degradation` 或设置 `csi_degradation.enabled: false`
- **THEN** dataset MUST 返回与当前 clean CSI loader 等价的 `csi` 张量
- **AND** 现有 CSI-only 与 CSI fusion 配置 MUST 不需要修改即可继续运行

#### Scenario: 解析 medium profile
- **WHEN** dataset 配置 `csi_degradation.enabled: true` 且 `profile: medium`
- **THEN** 系统 MUST 解析出 SNR 10 dB、path dropout 20%、AoA/AoD noise 3 度、delay noise 0.5 ns、antenna phase error 10 度和 temporal shift choices `[-1, 0, 1]`
- **AND** 用户在 YAML 中显式覆盖的单项参数 MUST 优先生效

#### Scenario: 解析 hard profile
- **WHEN** dataset 配置 `csi_degradation.enabled: true` 且 `profile: hard`
- **THEN** 系统 MUST 解析出 SNR 5 dB、path dropout 30%、dominant path attenuation 0.5、AoA/AoD noise 5 度、delay noise 1 ns、antenna phase error 20 度和 temporal shift choices `[-2, -1, 0, 1, 2]`
- **AND** 系统 MUST 不把 severe 参数作为默认 profile

### Requirement: MMW path-level CSI 退化算子
当 CSI 文件包含 MMW path-level channel 字段时，系统 MUST 优先在 path-level 执行退化，再派生模型输入 CSI tensor。支持的退化 MUST 包括 complex gain AWGN、弱路径优先 path dropout、dominant path attenuation、delay noise 或 quantization、AoA/AoD angle noise 和 antenna phase calibration error。退化后输出 MUST 仍满足 `[Nsc, Nant, 2]` 或 `[T, Nsc, Nant, 2]` real/imag 契约，并且 MUST 只包含 finite `float32` 数值。

#### Scenario: path-level payload 执行退化后输出稳定张量
- **WHEN** MMW channel payload 包含 path gain、delay 和 AoA/AoD 字段且启用 medium 或 hard degradation
- **THEN** loader MUST 在派生等效 CSI 前应用已配置的 path-level 退化算子
- **AND** loader MUST 返回 dtype 为 `float32` 且末维为 real/imag 的 finite CSI 张量

#### Scenario: 弱路径优先 dropout
- **WHEN** 配置启用 `path_dropout_rate` 且 payload 中存在多条 path gain
- **THEN** 系统 MUST 依据 path power 使弱路径比强路径更容易被置零或移除
- **AND** 系统 MUST 不在未配置 dominant path removal 的情况下直接删除所有最强路径

#### Scenario: 主径衰减
- **WHEN** 配置启用 `dominant_path_attenuation: 0.5`
- **THEN** 系统 MUST 将每个样本或每个可识别 channel 的 dominant path complex gain 乘以 0.5
- **AND** 系统 MUST 保留该 path 的相位和非零结构

#### Scenario: tensor-only payload fallback
- **WHEN** CSI 文件只包含 complex channel tensor 而不包含 path-level delay 或 angle 字段
- **THEN** 系统 MUST 仍可应用 tensor-level AWGN 和 antenna phase error
- **AND** 系统 MUST 在 diagnostics 中记录无法执行的 path-level 算子

### Requirement: CSI temporal shift 不得引入未来信息
CSI temporal shift MUST 只作用于当前样本的历史 CSI 路径序列，不得读取或构造未来 CSI 输入。默认边界处理 MUST 使用 clamp 到最近历史帧，或使用显式配置的无信息填充值。

#### Scenario: 历史窗口内 temporal shift
- **WHEN** 历史 CSI 路径为 `csi1..csi8` 且采样到 shift `+1`
- **THEN** dataset MUST 只从这 8 个历史 CSI 路径中重排或边界填充得到模型输入
- **AND** dataset MUST 不读取 `future_beam*` 对应帧的 CSI

#### Scenario: temporal shift 边界可复现
- **WHEN** shift 后的索引越过历史窗口边界
- **THEN** 系统 MUST 按配置的 fill mode 进行边界处理
- **AND** diagnostics MUST 记录本样本实际使用的 shift 和 fill mode

### Requirement: CSI degradation 随机性与 diagnostics
系统 MUST 使用稳定随机种子生成 degradation 随机数，保证同一配置、split、样本 index 和 CSI 路径集合在重复运行中得到相同 degraded CSI。系统 MUST 能记录 degradation profile、seed、有效参数、skipped operators 和每个样本的 temporal shift 信息。

#### Scenario: 重复读取同一样本结果一致
- **WHEN** 使用相同 `csi_degradation.seed`、split 和样本 index 重复读取同一个 CSI 样本
- **THEN** dataset MUST 返回数值一致的 degraded CSI
- **AND** 多 worker dataloader 的读取顺序变化 MUST 不改变该样本的退化结果

#### Scenario: 记录退化诊断信息
- **WHEN** dataset 启用 return metadata 或运行入口写出 run metadata
- **THEN** metadata MUST 包含 degradation profile、resolved parameters 和 seed
- **AND** 如果某些 path-level 算子因 payload 字段缺失未执行，metadata 或 diagnostics MUST 记录 skipped operators

### Requirement: Degraded CSI 实验配置
项目 MUST 提供至少一个 CSI-only degraded 配置和至少一个包含 degraded CSI 的 fusion 示例配置。配置 MUST 使用现有 dataset、model 和 batch 契约，并 MUST 显式记录 degradation profile。

#### Scenario: CSI-only degraded 配置可加载
- **WHEN** 用户加载 CSI-only degraded no-KD 配置
- **THEN** 配置 MUST 设置 `experiment.task: csi`
- **AND** 配置 MUST 启用 `data.dataset.use_csi: true` 和 `data.dataset.csi_degradation`

#### Scenario: fusion degraded CSI 配置可加载
- **WHEN** 用户加载包含 CSI 与其它模态的 degraded fusion 配置
- **THEN** 配置 MUST 将 `csi` 包含在 fusion `modalities`
- **AND** dataset MUST 只对 CSI 输入应用 CSI degradation，不得改变其它模态张量或 future beam 标签
