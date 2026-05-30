## MODIFIED Requirements

### Requirement: Fusion canonical 配置语义
每个 canonical fusion 配置 MUST 使用固定模态顺序 `image`、`radar`、`gps`、`lidar`、`mmwave` 生成 slug，并 MUST 让 teacher 和 student 使用相同的 `modalities`。推荐/default fusion student 路线 MUST 使用 `cls_token_transformer_fusion` 作为混合方式；已退役的 CRAF、MARF、G2D 配置、overlay 和 alias MUST 不再作为可解析 canonical 或高级方法入口。同一 slug 的四种配置 MUST 只改变训练角色和 KD 模式，不得改变模态集合。canonical 配置语义 MUST 不依赖实体 YAML 文件是否存在。

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

#### Scenario: 退役高级方法配置不可解析
- **WHEN** 用户加载 CRAF、MARF 或 G2D 的实体 YAML、virtual alias 或 overlay recipe
- **THEN** 系统 MUST 拒绝该配置路径或方法名
- **AND** 系统 MUST 不生成等价配置或兼容重定向

### Requirement: Fusion image encoder 与 profile 校验
Fusion 模型构建 MUST 校验启用 image modality 时的 image encoder 和 image profile 是否匹配。该校验 MUST 覆盖当前保留的 fusion、token transformer fusion 和模块化 fusion 入口，或在不支持某配置的入口处给出明确错误。

#### Scenario: fusion 使用 RGB profile
- **WHEN** 用户为 `fusion_teacher`、`fusion_student` 或 token transformer fusion 配置 `image_profile: rgb_imagenet`
- **THEN** 系统 MUST 构建或要求 3 通道 image branch
- **AND** 错误信息 MUST 在通道数不匹配时说明期望和实际通道数

#### Scenario: ResNet-18 fusion 使用 RGB profile
- **WHEN** 用户在 fusion 中选择 ResNet-18 image encoder 且 image profile 为 `rgb_imagenet`
- **THEN** 系统 MUST 构建或运行该配置
- **AND** image batch MUST 具有 3 通道 RGB/ImageNet 输入

#### Scenario: 已退役 fusion 方法不参与 profile 校验
- **WHEN** 配置请求 CRAF 或 MARF 风格 fusion
- **THEN** 系统 MUST 在 profile 校验前拒绝该模型类型
- **AND** 系统 MUST 不进入 CRAF/MARF 专属 image branch 构建逻辑

### Requirement: 默认 fusion no-KD 使用 CLS-token Transformer
推荐/default fusion no-KD 配置 MUST 使用 CLS-token Transformer fusion 作为混合方式。显式命名为 legacy 或 early-concat 的保留配置 MUST 保持其方法语义，不得被默认行为覆盖；已退役的 CRAF、MARF、G2D 或相关 ablation 配置 MUST 不再作为支持入口存在。

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

#### Scenario: 已退役方法不被默认配置保留
- **WHEN** 用户查找默认或高级 fusion no-KD 推荐入口
- **THEN** 项目 MUST 不再提供 CRAF、MARF、G2D 或其 ablation 配置作为推荐入口

## REMOVED Requirements

### Requirement: Fusion 配置选择 CRAF 模型
**Reason**: CRAF 架构已退役。
**Migration**: 使用 `cls_token_transformer_fusion`、`fusion_teacher/student` 或当前保留的 fusion baseline。

#### Scenario: CRAF fusion 配置不可用
- **WHEN** 用户在 fusion 配置中设置 `model.student.type: craf_fusion`
- **THEN** 系统 MUST 拒绝构建该模型
- **AND** 配置加载测试 MUST 不再要求 CRAF 示例配置成功

### Requirement: CRAF 配置复用 fusion 模态校验
**Reason**: CRAF 配置入口删除后，不再需要定义其模态校验复用语义。
**Migration**: 当前保留 fusion 模型继续复用标准 `modalities` 校验。

#### Scenario: CRAF 模态校验不再执行
- **WHEN** 配置请求 CRAF 模型
- **THEN** 系统 MUST 直接拒绝该模型类型
- **AND** 不再进入 CRAF 专属 token、gate 或 reliability 校验

### Requirement: CRAF canonical 与示例配置
**Reason**: CRAF 示例和 baseline 配置退役。
**Migration**: 使用当前保留的 canonical fusion 配置。

#### Scenario: CRAF 示例配置删除
- **WHEN** 用户查看 `configs/fusion/`
- **THEN** 项目 MUST 不再提供 CRAF 示例配置作为支持入口

### Requirement: CRAF 稳定化配置字段
**Reason**: CRAF 稳定化训练策略随 CRAF 删除。
**Migration**: 不提供兼容；新的稳定化策略需重新提出。

#### Scenario: CRAF 稳定化字段不可用
- **WHEN** 配置包含 CRAF warmup、counterfactual、ignore band 或 gate schedule 字段
- **THEN** 当前保留模型 MUST 不解释这些字段为有效行为

### Requirement: Teacher-prior CRAF 配置入口
**Reason**: teacher-prior CRAF Stage 2/3 和消融入口退役。
**Migration**: 使用当前保留的 fusion 训练配置。

#### Scenario: teacher-prior CRAF 配置删除
- **WHEN** 用户加载 teacher-prior CRAF Stage 2、Stage 3 或消融配置
- **THEN** 系统 MUST 拒绝该配置或报告文件不存在

### Requirement: CRAF gate 类型配置
**Reason**: CRAF gate 类型只服务于已删除 `craf_fusion`。
**Migration**: 无兼容迁移。

#### Scenario: CRAF gate 类型不可用
- **WHEN** 用户设置 CRAF `gate_type`
- **THEN** 系统 MUST 不构建 CRAF gate
- **AND** 错误 MUST 指向退役模型类型或未知字段

### Requirement: Fusion G2D 五模态配置入口
**Reason**: G2D 多模态蒸馏退役。
**Migration**: 使用普通 fusion no-KD、logits KD 或 RKD 配置。

#### Scenario: G2D 配置入口删除
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml`、`g2d_global` 或 `g2d_horizon`
- **THEN** 系统 MUST 不再解析为支持配置
- **AND** 项目 MUST 不保留 virtual alias 接管这些路径

### Requirement: Fusion student exposes modality features for G2D
**Reason**: 该输出适配只为 G2D feature KD 服务。
**Migration**: 当前模型仍可暴露自身 diagnostics，但 active specs 不再要求为 G2D 拆分 per-modality feature。

#### Scenario: G2D feature 输出不再要求
- **WHEN** fusion student forward 完成
- **THEN** 系统 MUST 不要求输出 G2D 专用 branch feature diagnostics
- **AND** G2D feature KD 测试 MUST 被删除

### Requirement: 高级 fusion 方法配置 overlay
**Reason**: 该 overlay 要求主要服务 CRAF、MARF、G2D 高级方法矩阵；这些方法已退役。
**Migration**: 当前保留的 canonical/virtual 配置继续由通用配置解析规格约束。

#### Scenario: 高级方法 overlay 删除
- **WHEN** 用户加载 CRAF、MARF 或 G2D method overlay
- **THEN** 系统 MUST 不再生成这些高级方法配置
- **AND** 测试 MUST 不再要求对应 overlay recipe 存在

### Requirement: 高级 fusion 实体 YAML 兼容
**Reason**: 不再保留 CRAF、MARF、G2D 实体 YAML 或 virtual alias 兼容。
**Migration**: 保留配置继续写出完整 `final_config.yaml` 和 `resolved_config.yaml`。

#### Scenario: 退役实体 YAML 不被 virtual 接管
- **WHEN** CRAF、MARF 或 G2D 实体 YAML 被删除
- **THEN** 配置加载器 MUST 不提供同名 virtual alias
- **AND** 用户引用该路径时 MUST 得到清晰错误
