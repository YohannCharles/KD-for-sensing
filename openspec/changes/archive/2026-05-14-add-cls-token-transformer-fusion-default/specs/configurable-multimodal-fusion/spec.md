## MODIFIED Requirements

### Requirement: Fusion canonical 配置语义
每个 canonical fusion 配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar`、`mmwave` 生成 slug，并 MUST 让 teacher 和 student 使用相同的 `modalities`。默认/recommended fusion student 路线 MUST 使用 `cls_token_transformer_fusion` 作为混合方式；legacy early-concat、CRAF、MARF 和其它高级方法 MUST 通过显式配置路径或 overlay 选择。同一 slug 的四种配置 MUST 只改变训练角色和 KD 模式，不得改变模态集合。canonical 配置语义 MUST 不依赖实体 YAML 文件是否存在。

#### Scenario: fusion teacher no-KD 配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_teacher_no_kd.yaml`
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 将被训练主模型配置为 `fusion_teacher` 或明确命名的 teacher baseline
- **AND** `model.teacher.modalities` 与 `model.student.modalities` MUST 等于 slug 表示的模态集合

#### Scenario: fusion student no-KD 默认配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_student_no_kd.yaml` 或推荐/default fusion no-KD student 配置
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 将被训练主模型配置为 `cls_token_transformer_fusion`
- **AND** `model.teacher.modalities` 与 `model.student.modalities` MUST 等于 slug 表示的模态集合

#### Scenario: fusion logits KD 默认配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_logits_kd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: logits_kd`
- **AND** 配置 MUST 构建 frozen `fusion_teacher` 或明确命名的 teacher baseline
- **AND** 配置 MUST 构建可训练 `cls_token_transformer_fusion`
- **AND** teacher 和 student 的 `modalities` MUST 相同
- **AND** 配置 MUST 默认解析同 slug 的 canonical teacher no-KD 输出中的 `best.pth`

#### Scenario: fusion RKD 默认配置
- **WHEN** 开发者加载 `configs/fusion/<slug>_rkd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: rkd`
- **AND** 配置 MUST 构建 frozen `fusion_teacher` 或明确命名的 teacher baseline
- **AND** 配置 MUST 构建可训练 `cls_token_transformer_fusion`
- **AND** teacher 和 student 的 `modalities` MUST 相同
- **AND** 配置 MUST 提供 RKD 参数并默认解析同 slug 的 canonical teacher no-KD 输出中的 `best.pth`

## ADDED Requirements

### Requirement: 默认 fusion no-KD 使用 CLS-token Transformer
推荐/default fusion no-KD 配置 MUST 使用 CLS-token Transformer fusion 作为混合方式。显式命名为 legacy、early-concat、CRAF、MARF、G2D 或 ablation 的配置 MUST 保持其方法语义，不得被默认行为覆盖。

#### Scenario: 加载五模态默认 fusion no-KD
- **WHEN** 用户加载推荐的五模态 fusion no-KD 配置
- **THEN** 配置 MUST 启用 `["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** `model.student.type` MUST 为 `cls_token_transformer_fusion`
- **AND** 配置 MUST 设置 CLS-token Transformer 所需的 `d_model`、`num_heads`、`num_layers` 或等价默认值

#### Scenario: 加载双模态默认 fusion no-KD
- **WHEN** 用户加载推荐的双模态 fusion no-KD 配置
- **THEN** 配置 MUST 使用 slug 表示的两个模态
- **AND** `model.student.type` MUST 为 `cls_token_transformer_fusion`
- **AND** dataset 字段 MUST 只启用该组合需要的模态数据

#### Scenario: 显式 early-concat baseline 不被覆盖
- **WHEN** 用户加载显式 early-concat、legacy fusion 或模块化 `early_concat_gru` 配置
- **THEN** 系统 MUST 保持该配置声明的模型类型和 representation core
- **AND** 系统 MUST 不将其静默改写为 `cls_token_transformer_fusion`

### Requirement: CLS-token Transformer 配置复用 fusion 数据字段
CLS-token Transformer fusion 配置 MUST 复用现有 fusion 数据字段和模态启用语义。启用 GPS、LiDAR 或 mmWave 时，配置 MUST 使用与其它 fusion 配置一致的数据字段、归一化和输入准备逻辑。

#### Scenario: 启用 GPS
- **WHEN** CLS-token Transformer fusion 配置的 `modalities` 包含 `gps`
- **THEN** 配置 MUST 设置 `data.dataset.use_gps: true`
- **AND** 配置 MUST 设置 `gps_feature_mode: relative_polar`
- **AND** `gps_input_size` MUST 为 3

#### Scenario: 启用 LiDAR
- **WHEN** CLS-token Transformer fusion 配置的 `modalities` 包含 `lidar`
- **THEN** 配置 MUST 设置 `data.dataset.use_lidar: true`
- **AND** 配置 MUST 沿用 LiDAR BEV 默认字段、缓存和内存有界归一化语义
- **AND** 模型 `lidar_channels` MUST 与 LiDAR BEV 输入通道一致

#### Scenario: 启用 mmWave
- **WHEN** CLS-token Transformer fusion 配置的 `modalities` 包含 `mmwave`
- **THEN** 配置 MUST 设置 `data.dataset.use_mmwave: true`
- **AND** 配置 MUST 设置 `mmwave_normalize: true`
- **AND** `mmwave_input_size` MUST 为 64
