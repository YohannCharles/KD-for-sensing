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
论文复现、官方协议包装或多阶段训练流程 MUST 放在 `src/kd_sensing/baselines/<family>/`、包内 CLI 或当前 allowlist 中的薄脚本入口。Workflow baseline MUST 明确标记与通用可训练 baseline 的区别，并 MUST 复用现有数据/metric/产物边界；不得复制长期通用训练循环或恢复退役入口。

#### Scenario: 论文复现 baseline 使用 workflow 路径
- **WHEN** baseline 包含官方源码审计、多阶段训练、feature cache、特殊 metric 或 Table 风格报告
- **THEN** 实现 MUST 使用 `kd_sensing.baselines.<family>` workflow、包内 CLI 或当前 allowlist 中的薄脚本入口
- **AND** 文档和 metadata MUST 标记其为 paper/workflow baseline，而不是普通 `modular_sequence` baseline

#### Scenario: workflow baseline 不恢复旧入口
- **WHEN** 新增 workflow baseline 需要命令入口
- **THEN** 入口 MUST 是包内 CLI 或已登记生命周期的薄 alias
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
- **WHEN** standard CNN+GPS、Image-AE+GPS、GPS-only 或不声明 reliability-aware 的模型运行
- **THEN** 新增 reliability metadata MUST 不成为必需 forward 输入
- **AND** 模型可比性 metadata MUST 能区分其未消费 reliability metadata
