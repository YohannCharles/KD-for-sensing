# model-architecture-extension-contract Specification

## Purpose
定义新增 baseline 与模型架构扩展的分类路径、模块化组件优先策略、整模型例外条件、workflow/paper reproduction 放置边界、metadata 可审计性和共享 batch/runtime 护栏。
## Requirements
### Requirement: Baseline 扩展路径分类
系统 MUST 将新增 baseline 或模型架构扩展划分为 config-only baseline、component baseline、whole-model exception 和 workflow/paper reproduction 四类路径。普通 supervised/adaptation baseline MUST 默认使用 config-only 或 component baseline；只有在 OpenSpec artifact 明确说明原因时，才可使用 whole-model exception 或 workflow/paper reproduction 路径。

#### Scenario: 新增普通 baseline 使用默认路径
- **WHEN** 开发者新增一个可复用现有训练、评估和 batch runtime 的 image、GPS、LiDAR、mmWave、CSI 或 fusion baseline
- **THEN** 该 baseline MUST 通过 `modular_sequence` 配置、virtual recipe 或新增 encoder/projector/core/head 组件表达
- **AND** 实现 MUST 不复制训练循环、dataset 解析或专用 batch forward 分支

#### Scenario: 新增复杂模型要求路径归类
- **WHEN** 开发者提出一个不能直接表达为现有 `modular_sequence` 组件组合的新模型
- **THEN** 对应 OpenSpec design 或 spec MUST 将其归类为 whole-model exception 或 workflow/paper reproduction
- **AND** 文档 MUST 说明不能使用 config-only 或 component baseline 的具体原因

### Requirement: 模块化组件优先
新增普通 baseline 的模型变化 MUST 优先落在 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 或 `HEADS` 子组件中。新增组件 MUST 保持现有输入/输出契约：encoder 输出 `[B, T, D_raw]`，projector 输出 `[B, T, d_model]`，core 输出可被 head 消费，head 输出与 `ModelOutput` 适配逻辑兼容。

#### Scenario: 新增视觉 encoder
- **WHEN** baseline 只替换 image 表征提取方式
- **THEN** 实现 MUST 新增或复用 `ENCODERS` 组件并通过 `model.primary.encoders.image.type` 选择
- **AND** 训练和评估入口 MUST 不新增 image 专用脚本或新的 dataset 字段

#### Scenario: 新增 fusion core
- **WHEN** baseline 只改变多模态融合或时序建模方式
- **THEN** 实现 MUST 新增或复用 `REPRESENTATION_CORES` 或等价可组合窄组件
- **AND** 配置 MUST 通过 `model.primary.representation_core.type` 或明确的窄组件字段选择该行为

### Requirement: Whole-model exception 契约
新增完整 `MODELS.register(...)` 可训练模型 MUST 是显式例外。该例外 MUST 提供 OpenSpec 设计理由、注册名、配置入口、支持模态、forward 输入、输出契约、metadata 契约和 focused tests。例外模型 MUST 复用 `engine.batch`、`engine.runtime` 和 `ModelOutput` 适配路径，不得要求训练循环新增模型专用分支。

#### Scenario: whole-model exception 可审计
- **WHEN** 新增模型文件注册新的 `MODELS` 名称
- **THEN** change artifact MUST 说明该模型为何不能只作为 encoder/projector/core/head 组件实现
- **AND** tasks MUST 包含 registry 构建、synthetic forward、output adaptation、metadata 和配置加载测试

#### Scenario: whole-model exception 不复制 runtime
- **WHEN** whole-model exception 被用于训练或评估
- **THEN** batch 输入 MUST 由共享 `prepare_task_inputs` 或等价共享 runtime 准备
- **AND** 模型输出 MUST 能被 `adapt_model_output` 消费

### Requirement: Workflow baseline 边界
论文复现、官方协议包装或多阶段训练流程 MUST 放在 `src/kd_sensing/baselines/<family>/`、包内 CLI 或 package console script 中。Workflow baseline MUST 明确标记与通用可训练 baseline 的区别，并 MUST 复用现有数据/metric/产物边界；不得复制长期通用训练循环或恢复退役入口。

#### Scenario: 论文复现 baseline 使用 workflow 路径
- **WHEN** baseline 包含官方源码审计、多阶段训练、feature cache、特殊 metric 或 Table 风格报告
- **THEN** 实现 MUST 使用 `kd_sensing.baselines.<family>` workflow、包内 CLI 或 package console script
- **AND** 文档和 metadata MUST 标记其为 paper/workflow baseline，而不是普通 `modular_sequence` baseline

#### Scenario: workflow baseline 不恢复旧入口
- **WHEN** 新增 workflow baseline 需要命令入口
- **THEN** 入口 MUST 是包内 CLI 或 package console script
- **AND** 系统 MUST 不新增 root-level 旧式训练脚本、兼容聚合层或退役研究线入口

### Requirement: 模型训练策略 metadata
新增可训练模型、组件 baseline 或 workflow baseline MUST 提供可审计训练策略 metadata。对于 `modular_sequence`，metadata MUST 由模型聚合 encoder/projector/core/head 信息自动生成；whole-model exception 和 workflow baseline MUST 提供 `training_strategy_metadata()` 或等价 run metadata helper。

#### Scenario: modular_sequence metadata 聚合
- **WHEN** `modular_sequence` baseline 使用自定义 encoder 或 core
- **THEN** run metadata MUST 记录启用模态、encoder 类型、projector/core/head 类型和关键训练策略字段
- **AND** 若组件使用 checkpoint reuse、freeze policy 或 reliability metadata，该信息 MUST 可在 metadata 中审计

#### Scenario: 整模型 metadata 最小字段
- **WHEN** whole-model exception 完成构建或训练
- **THEN** metadata MUST 至少包含模型注册名、启用模态、架构类别、是否消费 reliability metadata、是否使用外部 checkpoint 或 freeze policy
- **AND** 缺少 metadata 的新可训练模型 MUST 被 focused test 或架构边界测试发现

### Requirement: Reliability 和 adaptive fusion 可组合
新增 reliability-aware、observability-aware、uncertainty-gated 或 adaptive fusion 行为 MUST 优先作为可组合组件或显式 opt-in helper 暴露。该行为 MUST 记录是否消费 reliability metadata，并 MUST 保持普通 baseline 可忽略新增 metadata。

#### Scenario: adaptive fusion 显式 opt-in
- **WHEN** 配置启用 observability-aware 或 reliability-aware fusion
- **THEN** batch runtime MUST 只向声明消费 metadata 的模型传递 reliability fields
- **AND** benchmark comparability 或 run metadata MUST 记录该模型消费了 reliability metadata

#### Scenario: 普通 baseline 不被污染
- **WHEN** standard Image ResNet+GPS、Image-AE+GPS、GPS-only 或不声明 reliability-aware 的模型运行
- **THEN** 新增 reliability metadata MUST 不成为必需 forward 输入
- **AND** 模型可比性 metadata MUST 能区分其未消费 reliability metadata

### Requirement: Geometry-prior route classification
Geometry-prior beam fusion MUST 默认归类为 component baseline。实现 MUST 优先通过现有 `modular_sequence`、encoder/projector/core/head 或窄 fusion component 表达。

#### Scenario: 使用 component baseline 路径
- **WHEN** 开发者实现 GPS geometry prior、logit fusion 或 DBA-aware head
- **THEN** 实现 MUST 落在可注册的窄组件、loss/objective helper 或 diagnostics helper 中
- **AND** 系统 MUST 不新增完整 `MODELS.register(...)` 例外，除非 design 另行记录不可组合原因

#### Scenario: whole-model exception 需要明确理由
- **WHEN** geometry-prior 实现需要新增完整模型注册名
- **THEN** OpenSpec design 或 spec MUST 说明为什么不能使用 component baseline
- **AND** tasks MUST 包含 registry build、synthetic forward、ModelOutput adaptation、metadata 和 architecture boundary tests

### Requirement: BEV-Fusion reproduction boundary
完整 BEV-Fusion 论文复现 MUST 作为 workflow/paper reproduction 处理，而不是混入当前 geometry-prior component baseline。

#### Scenario: BEV-lite component 允许
- **WHEN** 实现只加入 GPS prior map、angle prior 或轻量 spatial prior token
- **THEN** 系统 MAY 将其作为 component baseline 实现
- **AND** metadata MUST 标记为 geometry-prior 或 BEV-lite，而不是完整 BEV-Fusion reproduction

#### Scenario: 完整论文复现走 workflow 路径
- **WHEN** 实现包含 camera-to-BEV、LiDAR/radar/GPS BEV、多阶段 preprocessing、论文 Table 复现或专用 feature cache
- **THEN** 系统 MUST 将其归类为 workflow/paper reproduction
- **AND** 入口 MUST 位于包内 CLI 或 `src/kd_sensing/baselines/<family>/`，不得新增旧式根脚本

### Requirement: Geometry-prior training metadata
Geometry-prior baseline MUST 写出可审计训练策略 metadata，覆盖 geometry prior、fusion、loss、teacher guidance 和 curriculum。

#### Scenario: metadata 最小字段
- **WHEN** geometry-prior model 构建或训练完成
- **THEN** metadata MUST 包含 model_group、architecture category、enabled modalities、geometry prior mode、fusion mode、loss mode、teacher guidance mode、curriculum mode 和 reliability metadata consumption
- **AND** 缺少这些字段 MUST 被 focused tests 或 architecture boundary tests 捕获

#### Scenario: baseline comparability metadata
- **WHEN** geometry-prior candidate 与 Image ResNet+GPS 或 JEPA GPS-query baseline 聚合比较
- **THEN** metadata MUST 声明 split、sample_count、metric_profile、normalization artifact、difficulty digest、history window、GPS source window、prediction horizon、scene set、seed、distance metric 和 beam label space
- **AND** 任一 strict 字段 mismatch MUST 阻止 claim upgrade

### Requirement: 模型扩展可被架构摘要审计
新增 baseline、组件 baseline、whole-model exception 和 workflow/paper reproduction MUST 能被模型架构摘要能力审计。审计信息 MUST 覆盖模型注册名或候选 ID、架构类别、启用模态、组件组合、参数量、checkpoint/freeze 策略、reliability metadata 消费和比较口径来源。

#### Scenario: component baseline summary 兼容
- **WHEN** 开发者新增或替换 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 或 `HEADS` 子组件
- **THEN** 该组件 MUST 能通过统一模型架构摘要出现在对应 role 分组中
- **AND** 摘要 MUST 记录该组件的 registry type、class、total params 和 trainable params

#### Scenario: whole-model exception summary 兼容
- **WHEN** 开发者新增完整 `MODELS.register(...)` 的 whole-model exception
- **THEN** 该模型 MUST 提供 `training_strategy_metadata()` 或等价 metadata，使架构摘要能记录模型注册名、架构类别、启用模态、checkpoint/freeze 策略和 reliability metadata 消费
- **AND** 如果无法自动分组内部组件，摘要 MUST 至少保留正确 total/trainable 参数和 unknown component role

#### Scenario: workflow baseline summary 兼容
- **WHEN** workflow/paper reproduction 生成候选、run manifest 或 summary table
- **THEN** 其参数量和 compute proxy 字段 MUST 能映射到统一模型架构摘要 schema
- **AND** summary MUST 区分真实实例统计和声明候选 metadata

### Requirement: 参数比较口径可审计
模型架构扩展 MUST 明确参数比较口径。系统 MUST 区分 total params、trainable params、effective params、excluded params、image encoder params、visual/context encoder params 和 compute proxy。任何参数量声明 MUST 记录来源，避免把 manifest 估算误当作真实 module 统计。

#### Scenario: 真实模型参数来源
- **WHEN** 参数量来自已构建 `nn.Module`
- **THEN** summary MUST 标记来源为实际 module 统计
- **AND** 参数量 MUST 使用去重后的 `named_parameters()` 或等价机制计算

#### Scenario: manifest 候选参数来源
- **WHEN** 参数量来自 sweep manifest、candidate metadata 或设计期估算
- **THEN** summary MUST 标记来源为声明候选 metadata
- **AND** summary MUST 不把该参数量标记为实际 module 统计

#### Scenario: 语义排除参数可追踪
- **WHEN** 模型实例包含不参与 downstream forward 的参数组
- **THEN** summary MUST 保留 total params
- **AND** summary MUST 在 effective/excluded 字段中记录排除口径和原因

### Requirement: 新模型 focused tests 包含摘要覆盖
新增模型、encoder、representation core、whole-model exception 或 sweep 候选矩阵时，focused tests MUST 覆盖模型架构摘要的关键字段。测试 MUST 至少验证 registry/candidate ID、组件 role、参数量字段、metadata 合并和 warning 语义。

#### Scenario: 新 encoder 摘要测试
- **WHEN** change 新增一个 image encoder 或其它模态 encoder
- **THEN** tasks MUST 包含该 encoder 在 `modular_sequence` 中生成架构摘要的 focused test
- **AND** test MUST 验证该 encoder 的 registry type、组件路径和参数量字段

#### Scenario: 新 sweep 候选摘要测试
- **WHEN** change 新增 sweep 候选族或参数/compute controls
- **THEN** tasks MUST 包含候选摘要 fixture 或 summary table test
- **AND** test MUST 验证候选参数来源、total/trainable params、token count 和 compute proxy

### Requirement: Legacy whole-model baseline retirement
普通 supervised/adaptation baseline 的 legacy whole-model 注册名 MUST 可退役为 removed guard。退役后，canonical config 路径 MAY 保留原文件名和 run name，但 `model.primary.type` MUST 使用 `modular_sequence` 或明确保留的 whole-model exception。

#### Scenario: single-modality strong/lightweight 退役
- **WHEN** canonical image、radar、GPS、LiDAR 或 mmWave strong/lightweight/supervised 配置仍作为 current 入口保留
- **THEN** 配置 MUST 使用 `model.primary.type: modular_sequence`
- **AND** 对应旧 whole-model 注册名 MUST NOT 出现在 current `MODELS.list()` 中

#### Scenario: legacy whole-model 名称必须有迁移 guard
- **WHEN** 用户请求本 change 退役的旧 whole-model 注册名
- **THEN** registry MUST 抛出 removed guard 错误
- **AND** 错误信息 MUST 指出等价或推荐的 `modular_sequence` 组件组合

#### Scenario: whole-model exception 保持显式
- **WHEN** 模型仍以完整 `MODELS.register(...)` 形式保留
- **THEN** 该模型 MUST 属于 workflow/paper reproduction、current spec 明确能力或 active design 记录的 whole-model exception
- **AND** 模型 MUST 继续提供可审计 metadata、registry build test 和 forward/output adaptation 覆盖

### Requirement: Model registry inventory reflects lifecycle
人类可读模型架构目录和机器可读维护索引 MUST 区分 current、removed guard 和 deferred cleanup。退役名称不得与 current 名称混列。

#### Scenario: 架构目录只展示 current 模型
- **WHEN** 维护者查看 `docs/model_architecture_inventory.md`
- **THEN** current model/encoder/core/head 表格 MUST 不包含 removed guard 名称
- **AND** 退役名称 MUST 只出现在退役边界或 migration 表中

#### Scenario: allowlist 与 registry 同步
- **WHEN** 架构边界测试读取 `docs/maintainer_context_index.yaml` 的 model registration allowlist
- **THEN** allowlist MUST 只包含 current `MODELS` 注册名
- **AND** removed guard 名称 MUST 被单独记录或通过测试断言其不可构建

### Requirement: Physics-informed MMW whole-model exception
系统 MUST 将 `pinn_multimodal_beam` 登记为显式 whole-model exception。该模型 MUST 说明不能仅通过 `modular_sequence` encoder/core/head 表达的原因，MUST 复用共享 batch/runtime 和 `ModelOutput` 适配路径，MUST 提供训练策略 metadata、registry build、synthetic forward、loss/backward 和架构摘要 focused tests。该例外模型的可替换前端 MUST 优先复用现有 encoder registry；paper-style frontend MUST 使用模态 tokenizer 和共享 Transformer 表达多模态感知，不得新增重复整模型注册名。

#### Scenario: PINN 模型例外可构建
- **WHEN** 构建流程导入默认组件并解析 `model.primary.type: pinn_multimodal_beam`
- **THEN** `MODELS` registry MUST 返回对应模型实例
- **AND** 该模型 forward 输出 MUST 能被 `adapt_model_output` 消费
- **AND** 训练循环 MUST 不需要新增模型专用 forward 分支

#### Scenario: PINN 模型 metadata 最小字段
- **WHEN** `pinn_multimodal_beam` 被构建或训练
- **THEN** `training_strategy_metadata()` 或等价 metadata MUST 记录模型注册名、architecture category、enabled modalities、physics branch、array type、codebook source、loss weights 和 sensitive physical supervision usage
- **AND** metadata MUST 记录该模型是否消费 CSI、path label、beam power 或 reliability metadata

#### Scenario: 架构摘要覆盖 PINN 模型
- **WHEN** 模型架构摘要检查 `pinn_multimodal_beam`
- **THEN** summary MUST 记录 total/trainable params、注册名、whole-model exception 类别、启用模态和 physics branch 配置
- **AND** 如果内部 path head 或 channel synthesizer 无法自动分组，summary MUST 至少保留 unknown component role 和参数统计

#### Scenario: paper-style 前端复用现有组件
- **WHEN** `pinn_multimodal_beam` 启用 paper-style tokenizer frontend
- **THEN** image tokenizer MUST 复用 `jepa_context_image`
- **AND** CSI/RF、radar、lidar 和 GPS tokenizer MUST 优先复用现有 `ENCODERS` 组件或薄 wrapper
- **AND** 系统 MUST 不新增第二个完整 PINN 模型注册名来表达同一物理链路

#### Scenario: paper-style 前端 metadata 可审计
- **WHEN** paper-style tokenizer frontend 被构建
- **THEN** metadata MUST 记录 frontend type、tokenizer type per modality、shared Transformer 层数、hidden_dim、checkpoint/freeze policy 和 whether GPS context is consumed
- **AND** image tokenizer metadata MUST 明确 `uses_gps_context=false`
