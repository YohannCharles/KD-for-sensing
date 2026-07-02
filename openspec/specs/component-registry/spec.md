# component-registry Specification

## Purpose
Define the lightweight registry contract for models, datasets, losses, metrics, and preprocessors, including explicit default component import boundaries and extension guidance.
## Requirements
### Requirement: 组件注册表
项目 MUST 提供轻量组件注册表，用于注册和构建模型、数据集、损失函数、指标和预处理器。注册表 MUST 支持按字符串名称查询组件，并通过配置参数实例化组件。项目 MUST 不再提供 `DISTILLERS` registry 或默认 distiller 注册流程。

#### Scenario: 按名称构建模型
- **WHEN** 配置中指定一个已注册模型名称和初始化参数
- **THEN** 系统 MUST 返回对应模型实例，并将配置参数传入模型构造函数

#### Scenario: 按名称构建数据集
- **WHEN** 配置中指定一个已注册数据集名称和初始化参数
- **THEN** 系统 MUST 返回对应 dataset 实例，并能被 DataLoader 使用

### Requirement: 可扩展模型和模态
新增普通 strong、lightweight、backbone、head、radar、GPS、LiDAR、mmWave、CSI 或 fusion baseline 时，开发者 MUST 优先通过 `modular_sequence` 配置、virtual recipe 或新增 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES`、`HEADS` 子组件扩展系统，而不需要复制训练脚本或修改训练循环主体。新增完整 `MODELS` 注册名 MUST 作为 whole-model exception 或 workflow/paper reproduction 在 OpenSpec artifact 中说明原因。

#### Scenario: 新增 image-only lightweight baseline
- **WHEN** 开发者实现一个新的 image-only lightweight baseline
- **THEN** 用户 MUST 能通过 `model.primary.type: modular_sequence` 和 `model.primary.encoders.image.type` 选择该 baseline
- **AND** 实现 MUST 复用现有 image-only 训练流程

#### Scenario: 新增多模态 fusion baseline
- **WHEN** 开发者实现一个新的 image+radar 或 radar+GPS fusion baseline
- **THEN** 用户 MUST 能通过 `modular_sequence` 的 encoders、projectors、representation core 和 heads 配置表达该 baseline
- **AND** 实现 MUST 不新增完整 `MODELS` 注册名，除非 active OpenSpec design 记录 whole-model exception 理由

#### Scenario: 新增 radar-only baseline
- **WHEN** 开发者实现 radar-only strong 或 lightweight baseline
- **THEN** 用户 MUST 能通过 `model.primary.encoders.radar.type: radar_cnn` 或新的 radar encoder 组件选择该行为
- **AND** 模型输出 MUST 继续兼容 `ModelOutput` 适配和 beam prediction loss/metric

### Requirement: 蒸馏扩展点已移除
项目删除 teacher-student KD 支持后，distiller 扩展点不再属于受支持架构。新监督损失 MUST 放入 loss、objective 或 training extension 模块；未来蒸馏方法 MUST 通过新的 OpenSpec change 重新定义。

#### Scenario: 选择 logits KD 被拒绝
- **WHEN** 配置中选择 logits KD
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 指向 supervised 或 adaptation 入口

#### Scenario: 选择 relational KD 被拒绝
- **WHEN** 配置中选择 relational KD
- **THEN** 系统 MUST 拒绝配置
- **AND** 系统 MUST 不构建 distiller

#### Scenario: registry 不暴露 distillers
- **WHEN** 开发者导入 `kd_sensing.registries`
- **THEN** 模块 MUST 不导出 `DISTILLERS`
- **AND** 默认组件导入 MUST 不导入 `kd_sensing.distillation.distillers`

### Requirement: 注册错误可诊断
注册表 MUST 对未知组件名称、重复注册名称和缺失必需参数提供明确错误信息，错误信息 MUST 包含注册表名称、请求的组件名称和可用组件列表或缺失字段。

#### Scenario: 请求未知组件
- **WHEN** 配置中引用未注册的模型、数据集、loss、metric 或 preprocessor 名称
- **THEN** 系统 MUST 抛出明确异常，并列出该注册表当前可用名称

#### Scenario: 重复注册组件名称
- **WHEN** 两个组件尝试注册到同一个注册表的相同名称
- **THEN** 系统 MUST 拒绝重复注册，并提示冲突名称和注册表类型

### Requirement: 组件发现文档
项目 MUST 在文档中说明如何查看可用组件、如何新增组件、如何在配置中引用组件，以及新增组件需要满足的输入输出约定。

#### Scenario: 按文档新增 metric
- **WHEN** 开发者按照 README 或扩展指南新增并注册一个 metric
- **THEN** 该 metric MUST 能被评估配置引用，并出现在评估结果输出中

### Requirement: 默认组件延迟导入
组件注册系统 MUST 保持注册表本身轻量可导入。导入 `kd_sensing.registries` MUST 不自动导入默认 dataset、model、preprocessor、diagnostics 或训练模块；默认组件注册 MUST 由显式注册导入函数或构建流程触发。

#### Scenario: 轻量导入 registry
- **WHEN** 开发者执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入默认 dataset、model 或 preprocessor 模块

#### Scenario: 构建前导入默认组件
- **WHEN** 构建流程需要通过 registry 构建已内置的 dataset、model、loss、metric 或 preprocessor
- **THEN** 构建流程 MUST 在查询 registry 前触发默认组件导入
- **AND** 已有配置中的 registry type MUST 继续可解析

### Requirement: 包级导出不扩大依赖面
包级 `__init__.py` 文件 MUST 避免 eager re-export 会引入重依赖或默认组件注册的符号。需要重依赖的功能 MUST 通过窄模块路径导入，或通过明确的延迟导入机制暴露。

#### Scenario: 导入 utils 包不触发 artifact registry
- **WHEN** 开发者执行 `import kd_sensing.utils`
- **THEN** 导入 MUST 不要求 dataset 场景、checkpoint registry 或 torch checkpoint 相关模块完成导入
- **AND** 路径和 seed 等轻量工具 MUST 仍可通过窄路径导入

#### Scenario: 显式导入 artifact registry
- **WHEN** 训练或评估代码需要 checkpoint registry 功能
- **THEN** 代码 MUST 从 `kd_sensing.utils.artifact_registry` 或等价窄入口导入
- **AND** checkpoint registry 行为 MUST 与变更前保持兼容

### Requirement: 注册发现文档区分轻量导入与组件注册
扩展文档 MUST 说明 registry 对象导入和默认组件注册是两个不同动作。文档 MUST 指导开发者在查看内置组件列表前显式导入默认组件或对应组件模块。

#### Scenario: 按文档查看内置模型
- **WHEN** 开发者按照扩展文档查看 `MODELS.list()`
- **THEN** 文档 MUST 要求先触发默认模型模块导入或调用默认组件导入函数
- **AND** 输出 MUST 包含内置模型注册名

#### Scenario: 按文档注册自定义组件
- **WHEN** 开发者在自定义模块中注册一个新组件
- **THEN** 文档 MUST 说明该模块需要在构建前被导入
- **AND** 系统 MUST 不通过扫描整个仓库隐式导入未知模块

### Requirement: 模块化模型组件注册
项目 MUST 通过现有组件注册边界暴露新的模块化序列模型及其可复用子组件。新增 image encoder、projector、representation core 和 head MUST 能通过配置名称构建，且不得要求训练脚本手写实例化逻辑。

#### Scenario: 按名称构建模块化序列模型
- **WHEN** 配置指定新的模块化序列模型注册名及其子组件配置
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建模型
- **AND** 构建参数 MUST 来自配置字段
- **AND** 训练循环 MUST 不需要为该注册名新增专用 forward 分支

#### Scenario: 按名称构建 ResNet-18 image encoder
- **WHEN** 模块化模型配置选择 `resnet18_imagenet_rgb` image encoder
- **THEN** 系统 MUST 通过注册表或明确 factory 构建该 encoder
- **AND** 未知 encoder 名称 MUST 使用现有 registry 错误风格报告可用名称

### Requirement: 默认组件导入包含新增组件
默认组件导入流程 MUST 注册 ResNet-18 image encoder、TinyViT image encoder、模块化序列模型和内置 core/head 组件，同时保持 registry 本身轻量可导入。导入 `kd_sensing.registries` MUST 不急切导入 torchvision、timm、dataset、训练器、checkpoint 文件、预训练权重接口或触发任何权重下载。

#### Scenario: 构建前导入默认组件
- **WHEN** 构建流程调用默认组件导入函数后再构建模块化序列模型
- **THEN** `MODELS` 注册表或对应子组件 registry MUST 包含新增注册名
- **AND** 用户配置中的新增注册名 MUST 可解析

#### Scenario: 轻量导入 registry 不触发 torchvision 或 TinyViT 权重
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import torchvision、timm 或 TinyViT 预训练权重接口
- **AND** 系统 MUST 不访问网络、不创建 checkpoint cache、不加载 TinyViT 权重

#### Scenario: 构建 TinyViT encoder 注册名
- **WHEN** 构建流程调用默认组件导入函数后查看 `ENCODERS.list()`
- **THEN** 输出 MUST 包含 `tinyvit_5m_scratch_rgb`、`tinyvit_5m_22k_rgb`、`tinyvit_11m_scratch_rgb` 和 `tinyvit_11m_22k_rgb`
- **AND** 系统 MUST 能通过 `ENCODERS.build()` 构建这些 TinyViT image encoder

#### Scenario: 未知 TinyViT 名称使用 registry 错误风格
- **WHEN** 用户请求不存在或拼写错误的 TinyViT encoder 注册名
- **THEN** 系统 MUST 使用现有 registry 错误风格抛出异常
- **AND** 错误信息 MUST 包含请求名称、registry 名称和可用 encoder 名称

### Requirement: 模块化组件错误可诊断
模块化模型构建失败时，系统 MUST 抛出包含组件类别、请求名称、相关模态和可用名称的清晰错误。shape 或 profile 不匹配错误 MUST 在构建或首次 forward 的早期暴露，并包含实际输入 shape。

#### Scenario: 未知 representation core
- **WHEN** 用户配置不存在的 `representation_core.type`
- **THEN** 系统 MUST 拒绝构建模块化序列模型
- **AND** 错误信息 MUST 包含请求的 core 名称和可用 core 名称

#### Scenario: encoder 与 profile 不匹配
- **WHEN** 用户配置 `rgb_imagenet` profile 但 image encoder 只支持 1 通道输入
- **THEN** 系统 MUST 拒绝构建或首次 forward
- **AND** 错误信息 MUST 包含 image profile、encoder 名称、期望通道数和实际通道数

### Requirement: Registry 只暴露 canonical 入口
组件注册表 MUST 只注册当前 canonical dataset、model、loss、metric 和 preprocessor 名称。已经由 canonical 名称替代的场景专用 dataset alias、旧模型类名 alias、legacy encoder alias，以及已退役的 CRAF、MARF、G2D、Multimodal-NF 和 distiller 入口 MUST 不再注册。

#### Scenario: dataset registry 不含场景专用 alias
- **WHEN** 构建流程导入默认 dataset 组件
- **THEN** `DATASETS` MUST 包含 `deepsense6g`
- **AND** `DATASETS` MUST 不包含 `scenario9`、`scenario31` 或 `scenario32`

#### Scenario: 旧 dataset alias 构建失败
- **WHEN** 用户配置 `the scene-9 dataset-type spelling`
- **THEN** registry 构建 MUST 拒绝该名称
- **AND** 错误信息 MUST 指向 `data.dataset.type: deepsense6g` 和 `data.dataset.scene: 9`

#### Scenario: 旧研究线入口构建失败
- **WHEN** 用户请求构建 `craf_fusion`、`marf_fusion`、`distillation.type: g2d` 或 `data.dataset.type: multimodal_nf`
- **THEN** registry 或配置构建 MUST 拒绝该名称
- **AND** 系统 MUST 不通过 deprecated alias、overlay 或兼容 facade 重定向到其它实现

#### Scenario: 旧模型类名 alias 不再导出
- **WHEN** 开发者从模型模块导入旧 fusion 类名 alias
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向当前公开类名或 canonical 模型注册名

### Requirement: 默认组件导入登记新增模型组件
新增内置模型子组件或整模型例外 MUST 被默认组件导入流程显式登记，同时保持 `kd_sensing.registries` 轻量可导入。默认组件导入 MUST 不通过兼容 facade、仓库扫描或旧聚合模块发现组件。

#### Scenario: 新组件可通过默认导入发现
- **WHEN** 构建流程调用 `import_default_components()` 后查询对应 registry
- **THEN** 新增内置 encoder/projector/core/head 或例外模型注册名 MUST 出现在 registry 列表中
- **AND** 仅导入 `kd_sensing.registries` MUST 不 eager import dataset、trainer、torchvision 权重接口或 checkpoint 文件

### Requirement: 轻量导入边界
项目 MUST 区分轻量基础模块和重依赖运行模块。导入配置加载、路径解析、场景元数据和模态契约时，系统 MUST 不导入 dataset、model、diagnostics、训练循环或需要 pandas、scipy、skimage、matplotlib 的模块。

#### Scenario: 缺少数据依赖时加载配置模块
- **WHEN** Python 环境可导入 `kd_sensing` 但缺少 pandas、scipy、skimage 或 matplotlib 中任一数据/可视化依赖
- **THEN** `import kd_sensing.config` MUST 成功
- **AND** 该导入 MUST 不触发 dataset 类、模型类或诊断渲染模块导入

#### Scenario: 只导入路径工具
- **WHEN** 开发者执行 `from kd_sensing.utils.paths import resolve_path`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 checkpoint registry、dataset 或模型模块

#### Scenario: 组件构建时才导入默认组件
- **WHEN** 训练或评估构建 dataset、model、loss、metric 或 preprocessor
- **THEN** 系统 MUST 显式导入默认组件以完成注册
- **AND** 该默认组件导入边界 MUST 不影响轻量配置加载路径

### Requirement: 包级导入不得牵出重依赖
项目 MUST 保持包级公共 API 兼容，同时避免 `__init__.py` eager import 触发重依赖运行模块。导入某个具体子模块时，系统 MUST 不因为父包初始化而额外导入训练器、dataset、诊断渲染或大型第三方依赖。已退役的 G2D、CRAF、MARF 和 Multimodal-NF 子模块 MUST 不再作为轻量导入 smoke 的保留对象。

#### Scenario: 导入 engine 轻量子模块
- **WHEN** 开发者执行 `import kd_sensing.engine.model_output`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `kd_sensing.engine._builders_impl`
- **AND** 系统 MUST 不导入 `kd_sensing.data.transform_ops._legacy`
- **AND** 系统 MUST 不导入 `pandas` 或 `scipy`

#### Scenario: 导入 diagnostics 轻量子模块
- **WHEN** 开发者导入当前保留的 diagnostics 轻量 helper
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `kd_sensing.diagnostics.visualization.core`
- **AND** 系统 MUST 不导入 `matplotlib`
- **AND** 系统 MUST 不要求 `kd_sensing.diagnostics.g2d_diagnostics` 存在

#### Scenario: distillation 子包不再作为 smoke 对象
- **WHEN** 开发者运行轻量导入 smoke
- **THEN** 检查 MUST 不导入 `kd_sensing.distillation`
- **AND** 系统 MUST 不要求 `kd_sensing.distillation.g2d_smp` 存在

#### Scenario: 旧 viewer 公共符号不可访问
- **WHEN** 现有代码执行 `from kd_sensing.diagnostics import export_viewer_manifest`
- **THEN** 导入 MUST 失败
- **AND** 错误信息或架构测试 MUST 指向当前 JEPA visual analysis、GPS shortcut benchmark 或其它非 viewer 诊断入口

### Requirement: models 包级轻量导入
`kd_sensing.models` MUST 保持轻量可导入，但不再 MUST 维持所有历史 package-level 模型符号兼容。该包 MAY 只暴露明确保留的当前公共符号、package metadata 或轻量 helper；当前内部代码、文档和测试 MUST 优先从真实 owner 模块、registry/config 名称或 package CLI 访问模型能力。删除的历史别名和便利导出 MAY 直接产生普通 `ImportError` 或 `AttributeError`，除非本 change 明确保留某个迁移 guard。

#### Scenario: 轻量导入 models 包
- **WHEN** 开发者执行 `import kd_sensing.models`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入各模型实现模块、训练 runtime、dataset reader 或重依赖视觉/科学计算模块

#### Scenario: 当前模型符号使用 owner 路径
- **WHEN** 当前源码、README、docs 或 tests 需要引用模型实现类
- **THEN** 引用 MUST 使用真实 owner 模块、canonical registry 名称或配置构建路径
- **AND** 不得要求 `kd_sensing.models.__all__` 继续列出历史便利导出

#### Scenario: removed alias 不再强制兼容
- **WHEN** 现有外部代码访问已移除的模型别名或历史 package-level 导出
- **THEN** 系统 MAY 抛出普通导入或属性错误
- **AND** 只有仍被当前迁移文档明确覆盖的别名才需要清晰替代符号提示

### Requirement: import 治理必须保护轻量导入边界
项目 MUST 将 import 治理重点放在 eager import、公开 facade 回流、跨领域依赖和重依赖泄漏上。轻量配置、路径、registry、package init、public facade 和 thin CLI MUST 不因架构整理额外导入 dataset reader、model implementation、training runtime、matplotlib、pandas、scipy、skimage、checkpoint 或权重文件。

#### Scenario: 轻量模块导入
- **WHEN** 开发者导入 `kd_sensing.config`、`kd_sensing.registries`、路径工具、包级公共 API 或已登记轻量 helper
- **THEN** 导入 MUST 成功且不触发训练、数据读取、模型权重加载或重型可视化依赖

#### Scenario: facade 内部回流
- **WHEN** 内部源码新增对公开 facade 中已迁移 helper 的 import 或调用
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向对应窄模块或 owner 模块作为迁移路径
