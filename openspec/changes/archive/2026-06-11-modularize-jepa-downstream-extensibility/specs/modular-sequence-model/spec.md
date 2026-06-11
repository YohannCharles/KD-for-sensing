## ADDED Requirements

### Requirement: conditioned encoder 契约
模块化序列模型 MUST 正式支持 encoder 声明条件依赖。声明条件依赖的 encoder MUST 指明依赖模态、条件特征来源和 forward kwarg；未声明依赖的 encoder MUST 继续保持单原始输入调用语义。该能力 MUST 不改变普通 encoder 输出 `[B,T,D_raw]`、projector 输出 `[B,T,d_model]` 和 representation core 输入契约。

#### Scenario: projected 条件特征注入
- **WHEN** `modular_sequence` 启用 image 和 GPS，且 image encoder 声明需要 `gps` 的 projected condition feature
- **THEN** 系统 MUST 先编码并投影 GPS，得到 `[B,T,d_model]`
- **AND** 系统 MUST 按 encoder 声明的 kwarg 名称将该 feature 传给 image encoder
- **AND** image encoder 输出 MUST 继续是 `[B,T,D_raw]`

#### Scenario: encoded 条件特征注入
- **WHEN** encoder 声明需要某个依赖模态的 encoded condition feature
- **THEN** 系统 MUST 将该依赖模态 projector 前的 encoder 输出 `[B,T,D_raw]` 传给目标 encoder
- **AND** 系统 MUST 在 metadata 或错误信息中区分 encoded 与 projected 来源

#### Scenario: raw 条件特征注入
- **WHEN** encoder 显式声明需要 raw condition feature
- **THEN** 系统 MUST 将对应 raw batch tensor 传给该 encoder
- **AND** raw 条件路径 MUST 只在 encoder 明确声明时启用，普通 encoder MUST 不读取其它模态 raw batch

#### Scenario: 条件依赖模态未启用
- **WHEN** encoder 声明需要 `gps` condition feature，但 `model.primary.modalities` 未启用 GPS
- **THEN** 系统 MUST 拒绝构建或 forward
- **AND** 错误信息 MUST 指出缺失的条件模态和依赖该条件的 encoder

#### Scenario: 条件 feature batch/time 不一致
- **WHEN** 条件 feature 的 batch 或 time 维与被条件化 encoder 的输入不一致
- **THEN** 系统 MUST 抛出包含两个 shape 的清晰错误
- **AND** 系统 MUST 不静默广播、截断或重排时间维

#### Scenario: 循环依赖被拒绝
- **WHEN** 多个 encoder 的条件依赖形成循环或无法满足的依赖图
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 包含 pending modalities 和 unmet dependencies

#### Scenario: 普通 encoder 兼容
- **WHEN** encoder 未声明任何条件依赖
- **THEN** `ModularSequenceModel` MUST 继续使用单个原始模态 tensor 调用该 encoder
- **AND** 现有 image、radar、GPS、LiDAR、mmWave、CSI、coord 和 ray 配置 MUST 无需新增条件字段即可 forward

### Requirement: token-valued representation 预留边界
模块化序列模型 MAY 在后续 change 中支持 token-valued encoder 输出，但当前 supervised JEPA downstream 默认契约 MUST 仍保持 `[B,T,D]` image feature。任何输出 `[B,T,K,D]` 或 `[B,T,N,D]` 的新路径 MUST 显式声明 representation kind，并 MUST 不破坏现有 core/head 的 `[B,T,D]` 和 `[B,K,T,D]` 输入契约。

#### Scenario: 当前 JEPA downstream 默认帧级输出
- **WHEN** `jepa_context_image` 使用 mean 或 GPS-query pooler
- **THEN** image encoder 输出 MUST 默认为 `[B,T,D]`
- **AND** 现有 projector、representation core 和 beam head MUST 无需 token-valued 特殊处理

#### Scenario: token-valued 输出需显式声明
- **WHEN** 后续配置选择输出 `[B,T,K,D]` 或 `[B,T,N,D]` 的 JEPA downstream pooler
- **THEN** 配置 MUST 显式声明 token-valued representation kind
- **AND** 系统 MUST 只将该输出传给声明支持 token-valued 输入的 core 或 adapter
