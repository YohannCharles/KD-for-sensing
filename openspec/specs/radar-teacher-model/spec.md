# radar-teacher-model Specification

## Purpose
定义 radar teacher 模型结构、注册名和 checkpoint 兼容行为，确保雷达教师分支的训练产物可被评估与蒸馏流程复用。
## Requirements
### Requirement: Radar-only 输入准备
系统 MUST 提供 radar-only 输入准备路径，从 batch 中读取 `radar_ra` 和 `radar_da`，在 channel 维拼接为雷达模型输入，并按现有预测窗口规则补齐未来占位帧。

#### Scenario: 准备 radar-only batch
- **WHEN** 训练、验证或评估流程处理 `experiment.task: radar`
- **THEN** 系统 MUST 使用 `radar_ra` 和 `radar_da` 构造雷达输入
- **AND** 系统 MUST 不要求图像输入参与模型 forward

#### Scenario: 雷达预测窗口对齐
- **WHEN** `seq_length` 为 8 且 `num_pred` 为 3
- **THEN** radar-only 输入 MUST 包含最近 8 个雷达时隙和 2 个未来 zero padding 时隙
- **AND** 验证和损失计算 MUST 使用最后 `num_pred` 个输出时隙与 `[t+1, t+2, t+3]` 标签对齐
- **AND** 输出时隙对齐 MUST 不包含历史窗口最后一个 beam

### Requirement: RadarTeacher 蒸馏角色已移除
系统 MUST 不再支持在 radar-only KD 配置中将 `radar_teacher` 作为 frozen teacher。Radar 强模型只能作为 supervised primary model、评估模型或可被显式权重评估的 checkpoint 来源。

#### Scenario: 构建 radar strong 模型
- **WHEN** 配置指定 radar strong primary model
- **THEN** 系统 MUST 通过模型注册表构建 `RadarModalityNet`
- **AND** 训练流程 MUST 将其作为被优化的 primary model

#### Scenario: 旧 radar KD 模型配置被拒绝
- **WHEN** 配置同时指定 frozen radar teacher、radar student 和 `logits_kd` 或 `rkd`
- **THEN** 配置加载 MUST 失败
- **AND** 系统 MUST 不执行 teacher/student forward

### Requirement: Radar strong canonical 配置使用 modular_sequence
Radar strong 和 supervised canonical 配置 MUST 使用 `modular_sequence`、`radar_cnn` encoder、projector、`single_gru` representation core 和 `beam_head`，而不是旧 Radar whole-model 注册名。

#### Scenario: 构建 radar strong/supervised 配置
- **WHEN** 用户加载 `configs/radar/strong.yaml` 或 `configs/radar/supervised.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** `model.primary.encoders.radar.type` MUST 为 `radar_cnn`
- **AND** radar-only task runtime MUST 继续从 batch 中准备 radar 输入并适配 beam logits

### Requirement: Radar strong legacy names are removed
Radar teacher/strong legacy whole-model 注册名 MUST 被 removed guard 拒绝，并指向 modular radar baseline。

#### Scenario: 请求 radar strong legacy 注册名
- **WHEN** 用户请求 `radar_teacher` 或 `radar_strong`
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence + radar_cnn + single_gru`

