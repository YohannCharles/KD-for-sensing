## ADDED Requirements

### Requirement: mmWave canonical 配置使用 modular_sequence
mmWave strong、lightweight 和 supervised canonical 配置 MUST 使用 `modular_sequence`、`mmwave_mlp` encoder、projector、`single_gru` representation core 和 `beam_head`，而不是旧 mmWave whole-model 注册名。

#### Scenario: 构建 mmWave strong/supervised 配置
- **WHEN** 用户加载 `configs/mmwave/strong.yaml` 或 `configs/mmwave/supervised.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** `model.primary.encoders.mmwave.type` MUST 为 `mmwave_mlp`
- **AND** mmWave-only task runtime MUST 继续只准备 mmWave 输入

#### Scenario: 构建 mmWave lightweight 配置
- **WHEN** 用户加载 `configs/mmwave/lightweight.yaml`
- **THEN** 系统 MUST 构建 `modular_sequence` mmWave-only 模型
- **AND** lightweight 差异 MUST 通过配置参数表达

### Requirement: mmWave legacy model names are removed
mmWave legacy whole-model 注册名和 feature extractor `MODELS` 注册名 MUST 被 removed guard 拒绝。

#### Scenario: 请求 mmWave legacy 注册名
- **WHEN** 用户请求 `mmwave_teacher`、`mmwave_student`、`mmwave_strong`、`mmwave_lightweight` 或 `mmwave_feature_extractor`
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence + mmwave_mlp + single_gru`

## MODIFIED Requirements

### Requirement: MmWaveFeatureExtractor 结构
系统 MUST 提供 `MmWaveFeatureExtractor`，用于从 mmWave 64 维 receive-power 特征序列中提取每个时隙的固定长度 embedding。该 feature extractor MUST 接收形状为 `(batch, sequence, 64)` 的 mmWave 张量，并输出 `(batch, sequence, feature_size)`。该类 MAY 通过 `kd_sensing.models.mmwave` 或 `kd_sensing.models` 窄导入暴露，但 MUST NOT 作为 current `MODELS` 注册名暴露。

#### Scenario: mmWave feature extractor 前向输出
- **WHEN** `MmWaveFeatureExtractor` 接收形状为 `(B, T, 64)` 的 mmWave 输入
- **THEN** 输出 MUST 为形状 `(B, T, feature_size)` 的特征张量
- **AND** 输出 feature 维 MUST 等于构造参数 `n_feature` 或 `feature_size`

#### Scenario: mmWave feature extractor 不作为完整模型注册
- **WHEN** 开发者查看 current `MODELS.list()`
- **THEN** 输出 MUST NOT 包含 `mmwave_feature_extractor`
- **AND** 需要配置构建 mmWave encoder 时 MUST 使用 `ENCODERS` 中的 `mmwave_mlp`

## REMOVED Requirements

### Requirement: mmWave teacher 模型结构
**Reason**: mmWave strong baseline 不再作为 standalone whole-model 注册名维护。
**Migration**: 使用 `configs/mmwave/strong.yaml` 或 `configs/mmwave/supervised.yaml`，其主模型为 `modular_sequence`。

### Requirement: mmWave student 模型结构
**Reason**: mmWave lightweight baseline 不再作为 standalone whole-model 注册名维护。
**Migration**: 使用 `configs/mmwave/lightweight.yaml`，其主模型为 `modular_sequence`。

### Requirement: mmWave-only 基线配置
**Reason**: 该要求指定通过 `mmwave_teacher` / `mmwave_student` 构建主模型，已与 modular canonical 入口冲突。
**Migration**: 使用本 change 新增的 `mmWave canonical 配置使用 modular_sequence` 要求。

### Requirement: mmWave 单模态默认 GRU 层数
**Reason**: 默认 GRU 层数不再属于 `mmwave_teacher` / `mmwave_student` whole-model 契约。
**Migration**: 在 `configs/mmwave/{strong,lightweight,supervised}.yaml` 的 `representation_core.num_layers` 中维护。
