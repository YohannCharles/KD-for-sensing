## ADDED Requirements

### Requirement: 条件化 encoder 调用
模块化序列模型 MUST 支持 encoder 显式声明依赖其它模态条件特征的调用路径。声明依赖的 encoder MAY 接收同 batch/time 的已编码或已投影条件特征；未声明依赖的 encoder MUST 保持单输入调用语义。该能力 MUST 不改变 encoder 输出 `[B,T,D_raw]`、projector 输出 `[B,T,d_model]` 和 representation core 输入契约。

#### Scenario: image encoder 接收 projected GPS 条件
- **WHEN** `modular_sequence` 启用 image 和 GPS，且 image encoder 声明需要 `gps` projected condition feature
- **THEN** 系统 MUST 先得到 GPS projector 输出 `[B,T,d_model]`
- **AND** 系统 MUST 将该 GPS condition feature 传给 image encoder
- **AND** image encoder 输出 MUST 继续是 `[B,T,D_raw]`
- **AND** 后续 image projector 与 fusion core MUST 按既有契约运行

#### Scenario: 条件依赖模态未启用
- **WHEN** encoder 声明需要 `gps` condition feature，但 `model.primary.modalities` 未启用 GPS
- **THEN** 系统 MUST 拒绝构建或 forward
- **AND** 错误信息 MUST 指出缺失的条件模态和依赖该条件的 encoder

#### Scenario: 条件 feature batch/time 不一致
- **WHEN** 条件 feature 的 batch 或 time 维与被条件化 encoder 的输入不一致
- **THEN** 系统 MUST 抛出包含两个 shape 的清晰错误
- **AND** 系统 MUST 不静默广播、截断或重排时间维

#### Scenario: 普通 encoder 兼容
- **WHEN** encoder 未声明任何条件依赖
- **THEN** `ModularSequenceModel` MUST 继续使用单个原始模态 tensor 调用该 encoder
- **AND** 现有 image、radar、GPS、LiDAR、mmWave、CSI、coord 和 ray 配置 MUST 无需新增条件字段即可 forward
