# gps-modality-model Specification

## Purpose
定义 GPS teacher/student 模型、特征模式和配置兼容行为，确保 GPS 分支能在单模态与 fusion 训练中稳定复用。
## Requirements
### Requirement: GPS teacher 模型
系统 MUST 提供已注册的 `gps_teacher` 模型，用于 GPS-only beam prediction。该模型 MUST 接收 `[B, T, 3]` GPS-Rel-Polar 特征序列，使用 `GpsFeatureExtractor` 提取每个时隙 embedding，经过 LayerNorm、GRU temporal modeling、时序增强模块和 MLP classifier 后输出 beam logits。

#### Scenario: 构建 gps_teacher
- **WHEN** 配置中指定 `model.teacher.type: gps_teacher` 或 `model.student.type: gps_teacher`
- **THEN** 模型注册表 MUST 能构建 `gps_teacher`
- **AND** 构建参数 MUST 支持 `gps_input_size`、`feature_size`、`num_classes` 和 `gru_params`
- **AND** GPS-Rel-Polar 配置中的 `gps_input_size` MUST 为 3

#### Scenario: gps_teacher forward contract
- **WHEN** `gps_teacher` 接收形状为 `[B, T, 3]` 的 GPS 输入张量
- **THEN** 模型 MUST 返回 `(pred, input_features, output_features)`
- **AND** `pred` 的形状 MUST 为 `[B, T, num_classes]`
- **AND** `input_features` 的形状 MUST 为 `[B, T, feature_size]`
- **AND** `output_features` 的 batch 和 sequence 维度 MUST 与输入一致

### Requirement: GPS student 模型
系统 MUST 提供已注册的 `gps_student` 模型，用于 lightweight GPS-only beam prediction。该模型 MUST 接收 `[B, T, 3]` GPS-Rel-Polar 特征序列，使用比 teacher 更轻量的 `GpsFeatureExtractor` 或投影层、LayerNorm、GRU temporal modeling 和小型 classifier 输出 beam logits。默认 GPS student 配置 MUST 使用 `gru_params: [64, 64, 1]`。

#### Scenario: 构建 gps_student
- **WHEN** 配置中指定 `model.student.type: gps_student`
- **THEN** 模型注册表 MUST 能构建 `gps_student`
- **AND** 构建参数 MUST 支持 `gps_input_size`、`feature_size`、`num_classes`、`gru_params` 和可选宽度控制参数
- **AND** GPS-Rel-Polar 配置中的 `gps_input_size` MUST 为 3

#### Scenario: gps_student forward contract
- **WHEN** `gps_student` 接收形状为 `[B, T, 3]` 的 GPS 输入张量
- **THEN** 模型 MUST 返回 `(pred, input_features, output_features)`
- **AND** `pred` 的形状 MUST 为 `[B, T, num_classes]`
- **AND** `input_features` 的形状 MUST 为 `[B, T, feature_size]`
- **AND** `output_features` 的 batch 和 sequence 维度 MUST 与输入一致

#### Scenario: gps_student 默认 GRU 层数
- **WHEN** 用户通过默认 GPS student 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1

### Requirement: GPS 模型参数校验
GPS teacher 和 GPS student MUST 校验 `gru_params` 和输入维度配置。`gru_params` MUST 包含 `[input_size, hidden_size, num_layers]`，且 `input_size` MUST 等于 `feature_size`。

#### Scenario: 非法 gru_params 长度
- **WHEN** 用户用长度不是 3 的 `gru_params` 构建 `gps_teacher` 或 `gps_student`
- **THEN** 系统 MUST 抛出配置错误

#### Scenario: GRU 输入维度不匹配
- **WHEN** 用户配置的 `gru_params[0]` 不等于 `feature_size`
- **THEN** 系统 MUST 抛出包含实际维度和期望维度的错误

### Requirement: GPS-only 任务输入
训练、验证和评估流程 MUST 支持 `experiment.task: gps`。GPS-only 任务 MUST 只准备 GPS 输入和 label，不要求 image 或 radar 输入。

#### Scenario: GPS-only 训练 forward
- **WHEN** 用户通过训练入口运行 `experiment.task: gps` 的配置
- **THEN** 系统 MUST 从 batch 中读取 `gps`
- **AND** 系统 MUST 按预测窗口规则补齐未来 GPS 占位时隙
- **AND** GPS 输入特征维度 MUST 为 3
- **AND** 系统 MUST 调用 GPS 模型完成 forward

#### Scenario: GPS-only 评估 forward
- **WHEN** 用户通过评估入口运行 `experiment.task: gps` 的配置和 GPS 模型权重
- **THEN** 系统 MUST 构建配置指定的 GPS 模型并只使用 GPS 输入完成评估
- **AND** GPS 输入特征维度 MUST 为 3
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标

### Requirement: GPS KD 入口已移除
GPS-only 训练 MUST 不再支持 logits KD、RKD 或 distiller 运行时。旧 GPS KD 配置路径 MUST 在配置解析阶段失败，并引导用户使用 `configs/gps/strong.yaml`、`configs/gps/lightweight.yaml` 或 `configs/gps/supervised.yaml`。

#### Scenario: GPS logits KD 被拒绝
- **WHEN** 用户运行旧 GPS-only logits KD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 frozen GPS teacher 或 distiller

#### Scenario: GPS RKD 被拒绝
- **WHEN** 用户运行旧 GPS-only RKD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不计算关系蒸馏损失

### Requirement: GPS teacher 默认 GRU 层数
默认 GPS teacher 单模态配置 MUST 使用一层 GRU，以便与当前单模态配置、README 和测试保持一致。

#### Scenario: gps_teacher 默认 GRU 层数
- **WHEN** 用户通过默认 GPS teacher no-KD 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1

### Requirement: GPS 模型可导出 coarse anchor
GPS 模型系统 MUST 支持显式 opt-in 的 coarse anchor export profile。启用该 profile 时，GPS encoder 或 GPS-only 模型 MUST 能输出 coarse anchor 字段；未启用时现有 GPS teacher/student 契约 MUST 保持兼容。

#### Scenario: GPS teacher/student 默认契约不变
- **WHEN** 用户运行现有 GPS teacher 或 GPS student no-KD 配置且未启用 coarse anchor export
- **THEN** 模型 MUST 继续输出既有 beam logits、input features 和 output features
- **AND** 系统 MUST NOT 要求 coarse label、coarse loss 或 GPS anchor metadata

#### Scenario: 启用 GPS coarse anchor export
- **WHEN** 用户配置 GPS 模型 `coarse_anchor.enabled=true`
- **THEN** 模型或训练 wrapper MUST 输出 `coarse_logits`、`center_beam`、`confidence` 和可选 `beam_scores`
- **AND** 输出形状 MUST 满足 `gps-coarse-anchor-prediction` 能力定义的 anchor 契约
- **AND** run metadata MUST 记录 anchor source 为 `gps_neural_coarse` 或等价配置值

#### Scenario: GPS coarse head 参数可配置
- **WHEN** 用户构建 GPS coarse anchor 模型
- **THEN** 配置 MUST 支持 `group_size`、`num_classes`、coarse head hidden size、dropout 和 loss weights
- **AND** 系统 MUST 校验 `num_classes` 能被 `group_size` 整除
- **AND** 非法配置 MUST 抛出包含 `num_classes` 和 `group_size` 的清晰错误

### Requirement: MMW Town GPS v2 model registration
系统 MUST 提供显式 opt-in 的 MMW Town GPS v2 模型构建能力。该能力 MUST 与既有 `gps_teacher`、`gps_student` 序列模型分离，并 MUST 通过模型注册或 runner 内部构建入口支持 GPS MLP backbone、SceneAdapterV2 和 residual logits 组合。

#### Scenario: 既有 GPS teacher/student 默认不变
- **WHEN** 用户运行现有 GPS-only `gps_teacher` 或 `gps_student` 配置
- **THEN** 系统 MUST 继续接收 `[B, T, 3]` GPS-Rel-Polar 输入
- **AND** 系统 MUST NOT 要求 MMW Town GPS v2 feature、scene_id、theta 或 branch_id

#### Scenario: 构建 MMW Town GPS v2 模型
- **WHEN** 用户通过 v2 配置选择 MMW Town GPS v2 model
- **THEN** 系统 MUST 构建轻量 MLP GPS backbone
- **AND** 系统 MUST 按配置构建 SceneAdapterV2 或 v1 baseline adapter
- **AND** forward 输出 MUST 至少包含 `logits`、`residual_logits`、可用的 `geo_logits` 和 adapter diagnostics

### Requirement: MMW Town GPS v2 feature validation
MMW Town GPS v2 模型 MUST 校验输入 feature 维度、scene id、num_beams 和 adapter 类型。非法配置 MUST 抛出包含字段名和可执行修复提示的错误。

#### Scenario: feature 维度不匹配
- **WHEN** v2 模型接收的 GPS feature 最后一维不等于配置声明的 input_dim
- **THEN** 系统 MUST 抛出配置或运行时错误
- **AND** 错误信息 MUST 包含实际维度、期望维度和 `model.input_dim`

#### Scenario: scene id 超出范围
- **WHEN** v2 adapter forward 收到超出已注册 scene 数的 scene_id
- **THEN** 系统 MUST 拒绝 forward
- **AND** 错误信息 MUST 包含 scene_id、num_scenes 和当前 scene mapping metadata

