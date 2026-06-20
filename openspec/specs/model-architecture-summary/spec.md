# model-architecture-summary Specification

## Purpose
TBD - created by archiving change add-model-architecture-summary. Update Purpose after archive.
## Requirements
### Requirement: 统一模型架构摘要 schema
系统 MUST 提供统一的模型架构摘要 schema，用于描述已构建模型实例、sweep 候选和既有 run artifact 的模型结构、组件组合、参数量、token/compute proxy、checkpoint/freeze 策略和 warning。摘要 MUST 是 JSON 可序列化对象，并 MUST 包含 `schema_version`、`source`、`model`、`parameters`、`components`、`warnings` 和 `comparability` 顶层字段。

#### Scenario: 摘要包含稳定顶层字段
- **WHEN** 用户对一个已构建模型实例生成架构摘要
- **THEN** 摘要 MUST 包含 `schema_version`、`source.kind`、`model.registry_type`、`model.class`、`parameters.total_params`、`parameters.trainable_params`、`parameters.frozen_params`、`components` 和 `warnings`
- **AND** 摘要 MUST 能被 `json.dumps()` 序列化

#### Scenario: 摘要区分来源
- **WHEN** 摘要来自真实 `nn.Module` 实例
- **THEN** `source.kind` MUST 为 `instance`
- **AND** `parameters.parameter_count_source` MUST 为 `actual_module` 或等价实际实例来源

#### Scenario: 候选摘要记录声明来源
- **WHEN** 摘要来自 JEPA sweep manifest 候选而未构建真实模型
- **THEN** `source.kind` MUST 为 `candidate`
- **AND** `parameters.parameter_count_source` MUST 为 `declared_candidate_metadata`

### Requirement: 实例级参数统计
系统 MUST 能从 PyTorch `nn.Module` 实例统计参数量。统计 MUST 去重共享参数，并 MUST 输出 total、trainable、frozen、effective 和 excluded 参数口径。统计 MUST 保留组件路径和模块 class，且 MUST 不依赖真实 dataset、DataLoader 或训练循环。

#### Scenario: 统计 total trainable frozen 参数
- **WHEN** summary helper 接收一个包含冻结和可训练参数的模型实例
- **THEN** `parameters.total_params` MUST 等于去重后所有参数 `numel()` 之和
- **AND** `parameters.trainable_params` MUST 等于 `requires_grad=True` 参数之和
- **AND** `parameters.frozen_params` MUST 等于 `requires_grad=False` 参数之和

#### Scenario: 统计语义排除参数
- **WHEN** 模型包含明确未参与 downstream forward 的参数组，例如未使用的 ImageNet classifier head
- **THEN** 摘要 MUST 在 `excluded_parameter_groups` 或组件级字段中记录该参数组
- **AND** `parameters.effective_params` MUST 能表达排除该参数组后的有效参数量

#### Scenario: 无 dataset 副作用
- **WHEN** 用户只对模型配置或模型实例生成架构摘要
- **THEN** 系统 MUST 不构建真实 dataset、不启动 DataLoader、不训练模型、不写 checkpoint
- **AND** 默认 MUST 不读取本地 `dataset/` 内容

### Requirement: 组件角色与参数分组
系统 MUST 按组件角色汇总参数量。对于 `modular_sequence`，摘要 MUST 至少识别 `encoders.<modality>`、`projectors.<modality>`、`representation_core` 和 `heads.<name>`；对于可识别的 JEPA/TinyViT/ResNet 视觉组件，摘要 MUST 支持 `image_encoder_params` 和 `visual_context_encoder_params` 等语义分组。

#### Scenario: modular_sequence 组件分组
- **WHEN** 用户对 image+GPS `modular_sequence` 模型生成摘要
- **THEN** `components` MUST 包含 image encoder、GPS encoder、representation core 和 beam head 对应条目
- **AND** 每个条目 MUST 包含 path、class、semantic_role、total_params 和 trainable_params

#### Scenario: 视觉 context encoder 分组
- **WHEN** 用户对 JEPA downstream image encoder 或 JEPA sweep 候选生成摘要
- **THEN** 摘要 MUST 能报告 image encoder params
- **AND** 摘要 MUST 能单独报告 visual/context encoder params 或等价视觉上下文编码器参数字段

#### Scenario: 未知组件保留总数
- **WHEN** 系统无法识别某个模块的语义角色
- **THEN** 摘要 MUST 将其标记为 `unknown_component` 或等价 role
- **AND** 该模块参数 MUST 仍计入模型总参数

### Requirement: 配置和 override 预检 warning
系统 MUST 在架构摘要中输出配置/构建预检 warning。warning MUST 包含机器可读 code、path、message 和 severity。系统 MUST 至少覆盖 encoder 专属选项不兼容、潜在 checkpoint 下载、声明参数与实际统计不一致、未知组件角色和未使用参数组。

#### Scenario: TinyViT 继承 ResNet stage 选项
- **WHEN** 用户将 ResNet 配置 override 为 TinyViT encoder 但仍保留 `unfreeze_stages: [layer4]`
- **THEN** 架构摘要或 preflight MUST 输出 `incompatible_encoder_option` warning 或构建前错误
- **AND** warning MUST 指出 TinyViT 可用 stage 与不兼容的配置路径

#### Scenario: 预训练权重潜在下载
- **WHEN** 用户摘要一个配置为 22k TinyViT 且未提供本地 `checkpoint_path` 的模型配置
- **THEN** 默认 summary build MUST 不下载权重
- **AND** 摘要 MUST 输出 `potential_checkpoint_download` warning 或要求用户显式允许下载

#### Scenario: 声明和实际参数不一致
- **WHEN** 候选 manifest 声明参数量与实际 build 统计差异超过配置阈值
- **THEN** 摘要 MUST 输出 `declared_vs_actual_param_mismatch` warning
- **AND** warning MUST 同时记录 declared 和 actual 参数量

### Requirement: 架构摘要 CLI 和输出格式
系统 MUST 提供薄 CLI 或等价包内入口来生成模型架构摘要。入口 MUST 支持从配置文件、override、sweep manifest 和既有 `startup_summary.json` 读取输入，并 MUST 支持 JSON、Markdown 和 CSV 中至少两种输出格式。默认输出 MUST 写入 stdout，只有用户显式指定路径时才写文件。

#### Scenario: 配置文件生成 Markdown summary
- **WHEN** 用户运行模型摘要 CLI 并传入 `--config` 和 `--format markdown`
- **THEN** 系统 MUST 构建或预检对应模型配置
- **AND** 系统 MUST 向 stdout 输出包含模型类型、启用模态、组件参数量和 warning 的 Markdown 表格或段落

#### Scenario: sweep manifest 生成 CSV summary
- **WHEN** 用户运行模型摘要 CLI 并传入 `--sweep-manifest` 和 CSV 输出格式
- **THEN** 系统 MUST 为 manifest 中选定候选生成同一 schema 的摘要行
- **AND** 输出 MUST 包含 `variant_id`、total params、trainable params、image encoder params、visual/context encoder params、token count 和 compute proxy 字段

#### Scenario: 显式输出路径遵守产物边界
- **WHEN** 用户为摘要 CLI 指定 `--output`
- **THEN** 系统 MUST 只写入用户指定路径
- **AND** 文档 MUST 推荐 ignored `outputs/analysis/model_architecture_summary/` 或其它显式本地产物目录

### Requirement: 训练 startup summary 兼容集成
系统 MUST 允许训练 startup summary 复用模型架构摘要能力，同时保持现有 startup summary 字段向后兼容。现有 `parameters.total_params`、`parameters.trainable_params` 和 `parameters.modules` 字段 MUST 保留；新增架构摘要字段 MUST 是 additive。

#### Scenario: 旧 startup 参数字段保留
- **WHEN** 训练流程写出 `startup_summary.json`
- **THEN** 文件 MUST 继续包含 `parameters.total_params`、`parameters.trainable_params` 和 `parameters.modules`
- **AND** 既有 TensorBoard startup scalar 写入 MUST 继续能读取这些字段

#### Scenario: 新架构摘要字段可用
- **WHEN** 训练流程启用或默认写入新架构摘要
- **THEN** `startup_summary.json` MUST 包含 `architecture_summary` 或等价新增字段
- **AND** 新字段 MUST 使用统一模型架构摘要 schema

