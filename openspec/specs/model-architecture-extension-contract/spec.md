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

### Requirement: AMBER full 默认使用 component baseline 路径
AMBER full architecture reproduction MUST 默认通过 `modular_sequence` 及其 encoder/projector/representation core/head/loss 组件实现。只有当 active design 证明该架构无法通过组件边界表达时，系统 MAY 使用 whole-model exception；该例外 MUST 提供额外 registry、forward、metadata、ModelOutput adaptation 和架构摘要测试。

#### Scenario: AMBER full 作为 representation core 构建
- **WHEN** AMBER full 变化集中在 fusion transformer、mask attention、CMA payload 或 beam head 输入表示
- **THEN** 实现 MUST 优先新增或扩展 `REPRESENTATION_CORES`、loss/objective helper 和配置
- **AND** 系统 MUST 不复制 dataset 解析、训练循环或专用 batch forward 分支

#### Scenario: whole-model exception 需要设计理由
- **WHEN** 实现者决定为 AMBER full 新增完整 `MODELS.register(...)` 名称
- **THEN** OpenSpec design 或后续 artifact MUST 说明不能使用 component baseline 的具体原因
- **AND** tasks MUST 包含 registry build、synthetic forward、`adapt_model_output`、metadata、architecture summary 和 architecture boundary tests

### Requirement: RBMA workflow extension path
RBMA、beam prototype alignment、full-to-partial teacher stabilization 和 pattern-balanced mask MUST 作为 U-MaskBeamJEPA opt-in 增强实现。除非后续 design 证明现有 whole-model exception 无法承载，系统 MUST 不新增第二个完整模型注册名来表达同一 workflow。

#### Scenario: 不新增重复 whole-model
- **WHEN** 实现 RBMA prototype KD workflow
- **THEN** 系统 MUST 复用 `u_mask_beam_jepa` 或现有 current owner
- **AND** 系统 MUST 不新增与 U-MaskBeamJEPA 语义重复的完整 `MODELS.register(...)` 名称

#### Scenario: 普通 baseline 不消费新增 metadata
- **WHEN** 普通 supervised、AMBER full local 或非 U-MaskBeamJEPA baseline 运行
- **THEN** reliability、prototype、full-to-partial teacher 和 pattern diagnostics MUST 不是必需 forward 输入
- **AND** 这些 baseline 的 metadata MUST 能声明未消费该 workflow metadata

### Requirement: RBMA workflow metadata
RBMA workflow MUST 写出可审计训练策略 metadata，覆盖 fusion type、mask sampler、prototype alignment、teacher stabilization、JEPA loss 状态、reliability metadata consumption 和 ablation id。

#### Scenario: metadata 最小字段
- **WHEN** RBMA workflow 模型或训练 run 被构建
- **THEN** metadata MUST 包含 model type、enabled modalities、fusion type、mask sampler、use_jepa_loss、use_beam_prototype_alignment、use_full_to_partial_kd 和 reliability metadata consumption
- **AND** 缺少这些字段 MUST 被 focused tests 或 architecture summary tests 捕获

#### Scenario: checkpoint teacher 状态可审计
- **WHEN** config 声明 `kd_teacher_mode`
- **THEN** metadata MUST 记录 teacher mode、teacher checkpoint provenance 或 pending reason
- **AND** checkpoint teacher 未实现时 MUST 不被记录为已启用成功

### Requirement: 训练方法扩展点边界
训练引擎 MUST 将后续仍被 OpenSpec 批准的方法所需的 teacher runtime、额外 loss、梯度后处理和 epoch diagnostics 接入点保持在明确模块边界内。`kd_sensing.engine.trainer` MUST 保持训练生命周期编排职责，不得作为方法特有 loss、teacher ensemble、counterfactual 或 subset-training 逻辑的主要实现位置。已退役的 G2D、CRAF 和 MARF 扩展模块 MUST 从 active code path 中删除。

#### Scenario: 新增训练方法不扩写主循环
- **WHEN** 开发者新增一个需要额外 loss 或 diagnostics 的训练方法
- **THEN** 主要实现 MUST 位于方法扩展模块及其测试中
- **AND** `kd_sensing.engine.trainer` 中的 epoch/batch 主循环 MUST 仅通过通用扩展点调用该方法

#### Scenario: 退役方法扩展模块删除
- **WHEN** 开发者查看训练期接入逻辑
- **THEN** 系统 MUST 不再保留 G2D、CRAF 或 MARF 的 teacher runtime、extra loss、subset/counterfactual forward 和 scalar diagnostics 作为 active 方法模块
- **AND** `trainer.py` MUST 不包含这些退役方法的大段私有 helper 实现

### Requirement: 共享任务 forward runtime
训练、验证、诊断预测和当前保留的 teacher runtime MUST 复用同一组任务 forward helper 来完成 batch 标准化、输入准备、model forward、输出适配和 future slot 选择。新增或修改模态输入准备、task forward 参数或强制模态 mask 行为时，变更 MUST 不要求在 trainer、validator 和当前诊断预测路径中重复修改分支逻辑。已退役 G2D teacher runtime 和 viewer prediction export 不再属于复用对象。

#### Scenario: 修改 fusion 输入准备只改 runtime helper
- **WHEN** 开发者调整 fusion task 的 `modalities` 输入准备或 force mask 透传逻辑
- **THEN** 主要变更 MUST 限定在共享 forward runtime 模块和测试
- **AND** 不需要分别修改 trainer、validator 和 viewer prediction 的 task 分支

#### Scenario: 验证路径复用训练输入契约
- **WHEN** 训练和验证使用同一个 fusion 配置运行
- **THEN** 两条路径 MUST 使用一致的 batch key、sequence padding、future slot 选择和 model output 适配语义
- **AND** validation metrics MUST 不依赖独立复制的 task forward 分支

### Requirement: 训练编排层保持窄职责
训练主循环 MUST 只协调 epoch、checkpoint、optimizer、scheduler、extension hook、validation 调用和运行产物写出。objective metric alias、available metric 计算、TensorBoard objective 字段、validation forward/loss/collect 和 canonical overlay 生成 MUST 位于对应窄模块。

#### Scenario: 新增 objective 不修改 trainer 主循环
- **WHEN** 开发者新增一个 prediction objective 并完成 objective metadata、loss 和 metrics 实现
- **THEN** 不得要求修改 trainer 主循环中的 early stopping alias 表、history 字段表或 TensorBoard objective 字段表
- **AND** trainer MUST 通过 objective metadata 自动记录该 objective 的 primary metric 和日志字段

#### Scenario: 修改 validation 指标不修改 trainer 主循环
- **WHEN** 开发者修复 validation pass 中某个 objective 指标的聚合方式
- **THEN** 变更 MUST 限定在 evaluation pass、objective metrics 或 evaluation metrics 模块
- **AND** 不需要编辑 trainer 主循环

### Requirement: 训练运行时编排职责拆分
训练引擎 MUST 将训练运行时状态、单 batch step、epoch metrics/history、checkpoint/sidecar、TensorBoard 和最终 artifact 写出拆到职责明确的窄模块或 helper。`kd_sensing.engine.trainer.train` MAY 保留为公开入口和顶层生命周期编排器，但 MUST 不继续直接承载这些细节的主要实现。

#### Scenario: batch step 逻辑位于窄模块
- **WHEN** 开发者查看训练中单 batch 的 prepare、forward、loss、backward 和 optimizer step 编排
- **THEN** 主要实现 MUST 位于 batch step runner 或等价窄模块
- **AND** `trainer.py` MUST 只负责调用该 runner 并消费其返回的 loss、diagnostics 和状态更新

#### Scenario: checkpoint 写出位于 checkpoint manager
- **WHEN** 开发者调整 `best.pth`、`best_top1.pth`、`last.pth`、sidecar 或 checkpoint registry archive 的写出逻辑
- **THEN** 主要变更 MUST 限定在 checkpoint manager 或等价窄模块
- **AND** 不需要编辑训练 batch 主循环

#### Scenario: 训练 artifact 写出位于 artifact writer
- **WHEN** 开发者调整 `train_log.json`、`training_outputs.npz`、`final_config.yaml`、训练曲线或 debug artifact 的写出逻辑
- **THEN** 主要变更 MUST 限定在 artifact writer、history recorder 或等价窄模块
- **AND** 不需要编辑模型 forward、KD loss 或 optimizer step 逻辑

### Requirement: 当前源码热点必须收敛为薄 facade
项目 MUST 优先防止当前仍保留的大型 workflow 或公开 orchestration 入口重新聚合职责。`src/kd_sensing/data/mmw/preparation.py`、evaluation pass、batch preparation、diagnostics benchmark owner 和训练主循环等当前热点 MUST 在 inventory 中记录拆分方向和预算；已退役的 Hist LOSO executor、viewer manifest 和 BGAM workflow MUST 不再作为当前热点或兼容 facade 要求。

#### Scenario: Hist executor 不作为当前 facade
- **WHEN** 开发者运行架构边界测试或审阅热点 inventory
- **THEN** 检查 MUST 不要求 `hist_beam_loso_execution.py` 存在
- **AND** 文档 MUST 不把 Hist executor 作为当前待拆热点、公开入口或兼容 facade

#### Scenario: MMW preparation facade 收敛
- **WHEN** 开发者查看或修改 MMW Town10 preparation 的配置解析、zip/input 审计、sensor/channel indexing、sequence split、beam power 派生、manifest 写出、report 或 proxy geometry 逻辑
- **THEN** 主要实现 MUST 位于 `preparation_config.py`、`preparation_audit.py`、`preparation_index.py`、`preparation_splits.py`、`preparation_beam_power.py`、`preparation_writers.py`、`preparation_geometry.py` 或等价窄模块
- **AND** `data/mmw/preparation.py` MUST 只承担公开 orchestration、现有公开 helper 的兼容导出和顶层参数编排
- **AND** 架构边界测试 MUST 拒绝把上述窄职责的大段 helper 重新实现到 `data/mmw/preparation.py`

### Requirement: 热点模块拆分边界
项目 MUST 为高变更频率的大型模块提供职责拆分路径。拆分后的窄模块 MUST 按 schema/constants、pure helper、reader、writer、orchestration 或 domain-specific adapter 组织，公开 facade MAY 保留兼容导出，但新内部代码 MUST 优先依赖窄模块。

#### Scenario: 新内部代码使用窄模块
- **WHEN** 开发者在训练、评估、预处理、诊断或 viewer 相关实现中新增代码
- **THEN** 新代码 MUST 优先从职责明确的窄模块 import
- **AND** 不得新增对仅用于兼容 re-export 的二级聚合模块的内部依赖

#### Scenario: 公开入口兼容
- **WHEN** 现有用户从公开 facade import 旧符号
- **THEN** 导入 MUST 继续成功，除非对应 change 明确声明 breaking change
- **AND** facade MUST 不触发比旧路径更重的 eager import

### Requirement: 拆分后轻量导入保持
热点模块拆分 MUST 不破坏现有轻量导入边界。schema、constants、objective metadata 查询、dataset descriptor 查询和 path helper 查询 MUST 不因为拆分而导入训练循环、dataset 实例、模型、大型可视化依赖或真实数据读取逻辑。

#### Scenario: objective schema 轻量导入
- **WHEN** 开发者导入 objective metadata 的 schema/registry 子模块
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入训练器、dataset、模型或 matplotlib

#### Scenario: dataset runtime schema 轻量导入
- **WHEN** 开发者查询 dataset descriptor 或 runtime schema helper
- **THEN** 查询 MUST 不打开 HDF5、CSV、image、LiDAR 或 checkpoint 文件
- **AND** 查询 MUST 不导入训练循环

### Requirement: 热点拆分必须保持公开行为兼容
热点模块拆分 MUST 保持现有公开 CLI、公开 import、manifest schema、run metadata、summary CSV/JSON、preparation artifact 命名、样本契约和默认路径策略兼容。拆分只允许改变内部模块组织，不得改变模型数值语义、数据 split 语义、beam label 语义或本地产物边界。

#### Scenario: 退役 Hist 产物只读保留
- **WHEN** 历史 HiST-Beam LOSO run metadata、summary JSON、checkpoint reuse metadata 或本地输出仍在 `outputs/` 中
- **THEN** 当前源码热点拆分 MUST 不要求这些 Hist artifact 可由当前 runner 继续生成
- **AND** cleanup/index 工具 MAY 将其作为历史或退役产物只读审计

#### Scenario: MMW preparation 产物兼容
- **WHEN** MMW preparation 拆分完成后运行 focused characterization tests
- **THEN** 现有 frame manifest、sequence split CSV、split metadata、beam power artifact、data availability report 和 report JSON 的关键字段 MUST 保持兼容
- **AND** 测试 MUST 覆盖公开 `prepare_town10_skybridge` 工作流和仍保留的公开 helper import

#### Scenario: 本地产物边界不随拆分改变
- **WHEN** 开发者实施热点拆分、运行 focused tests 或执行 CLI smoke
- **THEN** 变更 MUST 不包含对 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包或真实本地运行产物的删除、移动、压缩或重写
- **AND** 生成的临时验证产物 MUST 位于忽略规则覆盖范围内或测试临时目录中

### Requirement: JEPA downstream 扩展实现边界
项目 MUST 将 JEPA Stage 1 预训练主模型、JEPA downstream pooler/adapter、模块化 conditioned encoder、optimizer 参数组和 runtime metadata 维护在职责清晰的窄模块中。新增 JEPA downstream pooler 或 adapter MUST 不要求修改 dataset、训练主循环、checkpoint schema 或旧兼容入口。

#### Scenario: 新增 JEPA pooler 不修改训练主循环
- **WHEN** 开发者新增一个 JEPA downstream pooler
- **THEN** 变更 MUST 限定在 JEPA downstream pooler/adapter 模块、注册代码、配置和测试
- **AND** 不需要修改 `engine.trainer` 主循环或 supervised beam loss/metric 流程

#### Scenario: 新增 JEPA adapter 不修改 dataset
- **WHEN** 开发者新增一个 JEPA downstream adapter
- **THEN** 变更 MUST 不要求修改 DeepSense6G dataset、GPS transform、image preprocessing 或 DataLoader 构建逻辑
- **AND** adapter MUST 通过模型配置和 registry 接入

#### Scenario: 不恢复退役入口
- **WHEN** JEPA downstream extensibility change 落地
- **THEN** 系统 MUST 不新增 KD/distillation、HiST/Hist、Top8 selector、GPS residual、camera residual 或 legacy fusion 兼容入口
- **AND** 新能力 MUST 通过当前 `src/kd_sensing` 包结构和 registry 边界接入

### Requirement: optimizer 参数组构建位于 optim 模块
训练引擎 MUST 将参数组解析、模块名 pattern 匹配、重复匹配检测、未匹配参数处理和参数组 summary 维护在 `kd_sensing.engine.optim` 或等价窄模块中。训练主循环 MUST 只消费构建好的 optimizer 和 summary。

#### Scenario: 修改 JEPA 参数组不触碰 trainer
- **WHEN** 开发者调整 JEPA context encoder、GPS encoder、pooler、core 或 head 的参数组匹配规则
- **THEN** 主要变更 MUST 限定在 optimizer 构建模块及其测试
- **AND** 不需要编辑 `engine.trainer` 的 epoch 或 batch 编排逻辑

#### Scenario: 参数组 summary 写入现有日志路径
- **WHEN** 训练使用多个 optimizer 参数组
- **THEN** 现有训练日志和 TensorBoard scalar 映射 MUST 能记录每组 learning rate 和参数数量
- **AND** 未声明参数组时 MUST 保持现有单 `main` 组日志字段

### Requirement: runtime metadata 收集位于 run metadata 模块
JEPA downstream 结构 metadata MUST 由 `engine.run_metadata`、artifact writer 或等价窄模块收集。模型和子模块 MAY 暴露只读 metadata 方法；训练主循环 MUST 不手写 JEPA downstream 专属字段。

#### Scenario: 模型声明 metadata 被聚合
- **WHEN** `model.primary` 或其子模块提供 JEPA downstream training strategy metadata
- **THEN** runtime metadata 收集模块 MUST 将其写入 `final_config.yaml` 或等价运行 metadata
- **AND** metadata MUST 包含 pooler、adapter、checkpoint、freeze 和参数组摘要中的正式字段

#### Scenario: config fallback 兼容历史配置
- **WHEN** metadata 在模型构建前需要从配置生成
- **THEN** run metadata 模块 MAY 使用配置解析作为 fallback
- **AND** fallback MUST 与模型声明 metadata 的核心字段保持一致

### Requirement: 通用 baseline 与 workflow baseline 分层
项目 MUST 区分通用可训练 baseline 和 workflow/paper reproduction baseline。通用 baseline MUST 复用配置驱动训练、共享 batch/runtime 和模型 registry；workflow baseline MUST 只在需要官方协议、多阶段训练、特殊 metric 或报告产物时保留专用 orchestration，并 MUST 放在包内职责清晰的位置并记录生命周期、产物边界和 claim caveat。

#### Scenario: 通用 baseline 不修改训练循环
- **WHEN** 开发者新增普通 supervised/adaptation baseline
- **THEN** 变更 MUST 限定在配置、模型子组件、registry/default component 和 focused tests
- **AND** 不得为了该 baseline 修改 dataset 解析、训练主循环或公共 CLI 入口

#### Scenario: 论文复现 workflow 有边界
- **WHEN** 开发者新增包含官方协议、多阶段训练、特殊 metrics 或报告产物的 workflow baseline
- **THEN** 代码 MUST 位于 `src/kd_sensing/baselines/<family>/`、包内 CLI 或 package console script
- **AND** 文档 MUST 标记其不是普通 `modular_sequence` baseline，并说明输出只写入 ignored runtime artifact root

### Requirement: 新模型不得扩大入口表面
新增模型架构能力 MUST 不新增 root-level 旧脚本、兼容聚合层、退役研究线实体配置或绕过 `src/kd_sensing` 包结构的运行方式。若需要新增 CLI，MUST 是package console script 或包内 CLI，并同步 pyproject、README/docs、inventory 和架构边界测试。

#### Scenario: 新模型需要命令入口
- **WHEN** whole-model exception 或 workflow baseline 需要新的用户命令
- **THEN** 入口 MUST 通过 package console script 或包内 CLI 暴露
- **AND** 系统 MUST 不新增仓库根长期训练脚本或未登记脚本入口

### Requirement: 大 owner 保留必须有 accepted rationale 和验证命令
项目 MAY 保留较大的 owner 模块，但该 owner MUST 在维护索引或 inventory 中登记 `right-size-accepted` 或等价状态、accepted rationale、保留职责、验证命令和未来拆分触发条件。没有 accepted rationale 的超预算 owner MUST 被登记为 `split-next`、`monitor` 或 `defer-with-rationale`。

#### Scenario: 审计型 diagnostics owner 保留
- **WHEN** JEPA benchmark、visual analysis、run index 或 cleanup owner 因输出 schema 审计需要保持较大文件
- **THEN** 维护索引或 inventory MUST 记录该 owner 的职责边界、保留理由和 focused tests
- **AND** 新增实现 MUST NOT 回流到公开 facade 或轻量导入路径

#### Scenario: accepted owner 继续增长
- **WHEN** `right-size-accepted` owner 新增职责、超过既有 rationale 或触碰新的 public schema
- **THEN** 开发者 MUST 更新 accepted rationale 或将 owner 改为 split/monitor 状态
- **AND** 对应 focused tests MUST 覆盖新增职责

### Requirement: 新整模型注册受治理
组件注册系统 MUST 继续支持 `MODELS` 注册整模型，但新增整模型注册 MUST 被视为架构例外并纳入 OpenSpec、文档和测试护栏。新增普通 baseline MUST 优先注册 encoder/projector/representation core/head 子组件，而不是注册新的整模型。

#### Scenario: 子组件注册优先
- **WHEN** 新增模型能力可以表达为 encoder、projector、representation core 或 head
- **THEN** 实现 MUST 使用对应子组件 registry
- **AND** 不得仅为组合这些子组件而新增新的 `MODELS` 注册名

#### Scenario: 整模型注册需要例外说明
- **WHEN** 新增源码包含新的 `@MODELS.register(...)` 或等价模型注册
- **THEN** 对应 change MUST 提供 whole-model exception 理由
- **AND** focused tests MUST 覆盖 registry build、forward 输出、metadata 和轻量导入边界

### Requirement: 扩展文档区分默认和例外注册
组件发现和扩展文档 MUST 将新增 baseline 的默认路径描述为模块化配置或子组件注册。直接注册 `MODELS` 的示例 MUST 位于 whole-model exception 小节，并说明需要 OpenSpec 设计理由和 focused tests。

#### Scenario: 文档默认示例使用模块化组件
- **WHEN** 开发者阅读 Add a Model 或新增 baseline 指南
- **THEN** 首个示例 MUST 展示 `modular_sequence` 配置或子组件 registry
- **AND** 文档 MUST 不把直接 `@MODELS.register` 整模型作为普通 baseline 的默认建议

### Requirement: Baseline 源码与配置放置边界
系统 MUST 按行为而不是名称放置 baseline 相关实现。可由共享训练、评估、batch runtime 和模型架构摘要直接消费的模型能力 MUST 位于 `src/kd_sensing/models/` 或其窄子模块；论文复现、外部源码审计、多阶段训练、feature cache、特殊报告或 Table 风格 workflow MUST 位于 `src/kd_sensing/baselines/<family>/`、包内 CLI 或 package console script。本地可训练 baseline/control 配置 MUST 默认位于 `configs/fusion/` 或其实验子目录；外部复现、官方 artifact 审计或 source-audit manifest 配置 MUST 位于 `configs/baselines/`。

#### Scenario: 普通可训练 baseline 保持在模型组件路径
- **WHEN** baseline 能通过 `modular_sequence`、encoder/projector/representation core/head 或已有 whole-model exception 被共享训练 runtime 构建
- **THEN** 其模型实现 MUST 位于 `src/kd_sensing/models/` 或现有模型组件 owner
- **AND** 其本地训练配置 MUST 位于 `configs/fusion/`、`configs/fusion/experiments/` 或对应 current config family
- **AND** 系统 MUST 不因为名称包含 baseline 就把模型实现搬入 `src/kd_sensing/baselines/`

#### Scenario: Workflow baseline 使用 baseline package
- **WHEN** baseline 包含官方源码审计、多阶段训练、feature cache、专用 evaluation/report builder 或 Table 风格报告
- **THEN** workflow 实现 MUST 位于 `src/kd_sensing/baselines/<family>/`、包内 CLI 或 package console script
- **AND** 该 workflow MUST 不注册新的 `MODELS`、`ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 或 `HEADS` 名称来绕过模型组件边界

#### Scenario: 外部复现配置与本地训练配置分开
- **WHEN** 配置描述本仓库可训练 baseline/control
- **THEN** 配置 MUST 使用 `configs/fusion/` 或 current experiment config family
- **AND** 当配置描述外部 repo、官方 checkpoint、官方 prediction、source audit 或 blocked official reproduction 时，配置 MUST 使用 `configs/baselines/` 或明确的 diagnostics manifest 路径

### Requirement: New model extensions must avoid shared hotspot edits by default
新增普通 baseline、component baseline 或 workflow baseline MUST 默认不修改 dataset 主体、training loop、evaluation loop、batch runtime 主路由或 `ModularSequenceModel.forward` 主体。确需修改共享热点时，OpenSpec design/tasks MUST 说明原因、影响面、focused tests 和 public behavior compatibility。

#### Scenario: 普通 component baseline
- **WHEN** baseline 只替换 encoder、projector、core、head、loss、metadata 或 config recipe
- **THEN** 实现 MUST 限定在对应组件 owner、registry、config/spec/test
- **AND** 不得修改 dataset class 主体、trainer 主循环或 evaluation loop

#### Scenario: 共享契约确需扩展
- **WHEN** 新能力确实需要新增 batch field、model forward metadata 或 evaluation schema 字段
- **THEN** change artifact MUST 同步更新 modality/batch/runtime/model extension specs
- **AND** focused tests MUST 覆盖普通 baseline 忽略新增字段和 opt-in baseline 消费新增字段两种路径

### Requirement: Model architecture summary covers refactored components
重构或新增模型组件后，模型架构摘要 MUST 继续能审计 registry id、组件 role、参数量、trainable params、checkpoint/freeze policy、reliability metadata consumption 和 comparability metadata 来源。内部模块移动 MUST 不让 summary 回落为 unknown，除非 design 明确说明无法自动分组。

#### Scenario: 组件移动后摘要稳定
- **WHEN** encoder/core/head 或 whole-model exception owner 文件被移动、拆分或合并
- **THEN** architecture summary focused tests MUST 继续验证对应 registry type、class path、role 和参数量字段
- **AND** docs/model architecture inventory MUST 与 current registry surface 保持一致

### Requirement: Whole-model exceptions remain explicit after cleanup
删除 facade、合并 helper 或阶段化 forward 后，仍保留的 whole-model exception MUST 继续有 current spec、active design、inventory 或 focused test 说明。退役整模型 direct import、alias 或 removed wrapper 不得作为包结构保留对象。

#### Scenario: Whole-model exception audit
- **WHEN** cleanup 后扫描 `@MODELS.register(...)`
- **THEN** 每个完整模型注册名 MUST 能映射到 current capability、explicit exception 或 workflow/paper reproduction 边界
- **AND** 无 current 依据的旧整模型 class MUST 删除或从 registry surface 退出

