# model-architecture-summary Specification

## Purpose
定义模型架构摘要的统一 schema、参数统计口径、组件目录、sweep manifest 兼容和只读 CLI 边界，使维护者能比较当前模型与候选配置而不读取真实数据、不启动训练或写入 checkpoint。
## Requirements
### Requirement: 统一模型架构摘要 schema
系统 MUST 为已构建模型实例和 training startup artifact 提供统一架构摘要 schema。摘要 MUST 是 JSON 可序列化对象，并 MUST 包含 `schema_version`、`source`、`model`、`parameters`、`components`、`warnings` 和 `comparability` 顶层字段；系统 MUST 不要求同一 owner 解析 sweep candidate 或任意 config-only 输入。

#### Scenario: 摘要包含稳定顶层字段
- **WHEN** current consumer 对一个已构建模型实例生成架构摘要
- **THEN** 摘要 MUST 包含 `schema_version`、`source.kind`、`model.registry_type`、`model.class`、`parameters.total_params`、`parameters.trainable_params`、`parameters.frozen_params`、`components` 和 `warnings`
- **AND** 摘要 MUST 能被 `json.dumps()` 序列化

#### Scenario: 摘要来源是实际实例
- **WHEN** 摘要来自真实 `nn.Module` 实例
- **THEN** `source.kind` MUST 为 `instance`
- **AND** `parameters.parameter_count_source` MUST 为 `actual_module` 或等价实际实例来源

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
系统 MUST 按已构建模型的组件角色汇总参数量。对于 `modular_sequence`，摘要 MUST 至少识别 `encoders.<modality>`、`projectors.<modality>`、`representation_core` 和 `heads.<name>`；对于 current JEPA/TinyViT/ResNet 视觉组件，摘要 MUST 支持 image encoder 与 visual/context encoder 的实际实例参数分组。

#### Scenario: modular_sequence 组件分组
- **WHEN** current consumer 对 image+GPS `modular_sequence` 模型生成摘要
- **THEN** `components` MUST 包含 image encoder、GPS encoder、representation core 和 beam head 对应条目
- **AND** 每个条目 MUST 包含 path、class、semantic_role、total params 和 trainable params

#### Scenario: current visual context encoder 分组
- **WHEN** 已构建 current 模型包含 JEPA mean context、TinyViT 或 ResNet image encoder
- **THEN** 摘要 MUST 报告 image encoder params
- **AND** 能识别时 MUST 单独报告 visual/context encoder params

#### Scenario: 未知组件保留总数
- **WHEN** 系统无法识别某个模块的语义角色
- **THEN** 摘要 MUST 将其标记为 `unknown_component` 或等价 role
- **AND** 该模块参数 MUST 仍计入模型总参数

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

### Requirement: 默认实验记录 encoder 和 preprocessing profile
训练、验证和评估流程 MUST 在运行产物中记录 camera encoder 与 LiDAR preprocessing profile，使不同单模态 baseline 的结果可以横向比较。

#### Scenario: 记录 image encoder profile
- **WHEN** 一次 image-only 或包含 image 的 fusion 训练启动
- **THEN** final_config 或运行 metadata MUST 记录 image profile、image encoder 类型、是否使用预训练权重、权重名称、freeze 策略和实际可训练 stage

#### Scenario: 记录 LiDAR preprocessing profile
- **WHEN** 一次 LiDAR-only 或包含 LiDAR 的 fusion 训练启动
- **THEN** final_config 或运行 metadata MUST 记录 LiDAR normalization、cache、ROI、FoV、ground/background filter 和安全增强配置

### Requirement: Module trainability startup report
The training workflow MUST report trainable parameter counts by major module for debug runs. The report MUST distinguish CSI encoder, representation core, beam head and fusion modules when those modules exist.

#### Scenario: 打印模块参数统计
- **WHEN** a debug run builds the model
- **THEN** startup logs MUST include total parameter count and total trainable parameter count
- **AND** startup logs MUST include trainable parameter counts by CSI encoder, representation core, beam head and fusion module where present

#### Scenario: 发现模块无可训练参数
- **WHEN** a required trainable module has zero trainable parameters
- **THEN** startup logs MUST mark the module as suspicious
- **AND** the warning MUST include the module name and resolved model path

### Requirement: Resolved config artifact and startup summary
Every debug run MUST save the fully resolved configuration and print a startup summary of the fields needed to compare experiment variants. The summary MUST be generated after defaults, aliases and command-line overrides are applied.

#### Scenario: 保存 resolved config
- **WHEN** a debug run starts
- **THEN** the run output directory MUST contain `resolved_config.yaml` or an equivalent fully resolved config artifact
- **AND** the artifact MUST reflect defaults, generated config values, aliases and command-line overrides

#### Scenario: 打印关键配置摘要
- **WHEN** a debug run starts
- **THEN** startup logs MUST include modalities, dataset path, train/val split paths, `seq_len`, `num_pred`, `num_classes`, batch size, optimizer, learning rate, scheduler and max epochs
- **AND** startup logs MUST include model type, CSI encoder type, `d_model`, `delay_taps`, `view_fusion`, `use_internal_gru`, pilot estimator enabled/mode/SNR, `csi_hardening.enabled` and `csi_degradation.enabled`

### Requirement: Architecture summary 只保留 instance/startup supporting surface
模型架构摘要 MUST 作为 training startup、U-Mask/AMR/AMBER focused validation 和 Scene31-34 profile 的 supporting owner 保留。它 MUST 不再提供 standalone CLI、candidate sweep ingestion、config override preflight、Markdown/CSV renderer 或独立 report 产品面。

#### Scenario: Current consumer 继续读取 instance summary
- **WHEN** training startup 或 current focused test 对已构建模型生成摘要
- **THEN** helper MUST 返回稳定、JSON 可序列化的 instance parameter/component schema
- **AND** Scene31-34 profile MUST 继续能读取 startup artifact 中的 architecture summary

#### Scenario: Retired summary surface 不存在
- **WHEN** 用户请求旧 architecture-summary CLI、sweep manifest renderer 或 config-only preflight
- **THEN** 对应入口和实现 MUST 不属于 current surface
- **AND** 项目 MUST 不新增 replacement renderer 或 wrapper
