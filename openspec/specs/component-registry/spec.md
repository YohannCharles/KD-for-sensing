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

### Requirement: LiDAR 注册错误可诊断
LiDAR 相关注册错误 MUST 使用现有注册表错误风格，并在未知名称、重复名称或缺失必需参数时提供清晰错误信息。

#### Scenario: 请求未知 LiDAR 组件
- **WHEN** 配置中引用未注册的 LiDAR 模型或预处理器名称
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含请求的组件名称、注册表名称和可用组件列表

#### Scenario: LiDAR 构建参数缺失
- **WHEN** 配置中引用已注册 LiDAR 组件但缺少必需构造参数
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含缺失字段或原始构建错误

### Requirement: mmWave 注册错误可诊断
mmWave 相关注册错误 MUST 使用现有注册表错误风格，并在未知名称、重复名称或缺失必需参数时提供清晰错误信息。

#### Scenario: 请求未知 mmWave 组件
- **WHEN** 配置中引用未注册的 mmWave 模型或预处理器名称
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含请求的组件名称、注册表名称和可用组件列表

#### Scenario: mmWave 构建参数缺失
- **WHEN** 配置中引用已注册 mmWave 组件但缺少必需构造参数
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含缺失字段或原始构建错误

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

### Requirement: 默认组件导入不依赖兼容模块
默认组件导入流程 MUST 注册 canonical 内置组件，同时保持 registry 本身轻量可导入。默认组件导入 MUST 不通过 `scenario9.py`、`engine.builders`、`data.transforms` 或 `_legacy` 兼容模块完成。

#### Scenario: 导入默认 dataset 组件
- **WHEN** 构建流程调用默认组件导入函数后再构建 DeepSense6G dataset
- **THEN** 默认导入 MUST 加载场景中立 dataset 模块
- **AND** 系统 MUST 不导入 `kd_sensing.data.datasets.scenario9`

#### Scenario: registry 轻量导入
- **WHEN** 开发者执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 dataset、model、training、checkpoint 或兼容 facade 模块

### Requirement: 已删除组件错误可诊断
当用户引用已删除的兼容组件名称或退役研究线组件名称时，注册表错误 MUST 至少包含请求名称、registry 名称或可用名称上下文。对于仍有当前迁移价值的名称，错误 SHOULD 区分“未知名称”和“已删除名称”并给出迁移方向；对于完全退役且不再承诺兼容的历史名称，系统 MAY 使用普通 unknown-name 错误或集中退役说明替代长期 removed guard table。

#### Scenario: 已删除 dataset type
- **WHEN** 用户请求构建 `scenario9` dataset 且项目仍保留该迁移说明
- **THEN** 系统 MUST 抛出包含 `scenario9` 的错误
- **AND** 错误信息 MUST 说明该名称已删除并给出 `deepsense6g + scene` 配置示例

#### Scenario: 已删除模型 alias
- **WHEN** 用户请求旧 fusion 类名 alias 或已删除 image encoder alias，且该名称仍在 current migration table 中
- **THEN** 系统 MUST 抛出包含请求名称的错误
- **AND** 错误信息 MUST 列出当前支持的 canonical 注册名

#### Scenario: 已退役研究线组件
- **WHEN** 用户请求 `craf_fusion`、`marf_fusion`、`g2d` distiller 或 `multimodal_nf` dataset
- **THEN** 系统 MUST 拒绝构建
- **AND** 系统 MAY 报告为已退役名称或普通未知名称，但 MUST 不通过 deprecated alias、overlay 或兼容 facade 重定向到其它实现

### Requirement: CLS-token Transformer fusion 组件注册
项目 MUST 通过现有组件注册表暴露 CLS-token Transformer fusion 模型。新增模型 MUST 能通过 `MODELS` 注册表构建，并 MUST 复用现有 fusion 训练、验证和评估入口。

#### Scenario: 按名称构建 CLS-token Transformer fusion
- **WHEN** 配置指定 `type: cls_token_transformer_fusion`
- **THEN** 系统 MUST 通过 `MODELS` 注册表返回 CLS-token Transformer fusion 模型实例
- **AND** 构建参数 MUST 来自配置字段
- **AND** 模型 MUST 支持现有 fusion forward 输入键

#### Scenario: 注册错误可诊断
- **WHEN** 用户引用不存在或拼写错误的 CLS-token Transformer fusion 注册名
- **THEN** 系统 MUST 使用现有 registry 错误风格抛出异常
- **AND** 错误信息 MUST 包含请求名称和可用模型注册名

### Requirement: 默认组件导入包含 CLS-token Transformer fusion
默认组件导入流程 MUST 注册 CLS-token Transformer fusion 内置模型，同时保持 registry 本身轻量可导入。导入 `kd_sensing.registries` MUST 不急切导入 dataset、trainer、checkpoint 或重依赖运行模块。

#### Scenario: 构建流程导入默认组件
- **WHEN** 构建流程调用 `import_default_components()` 后再查询 `MODELS`
- **THEN** `MODELS` 注册表 MUST 包含 `cls_token_transformer_fusion`
- **AND** 系统 MUST 能通过配置构建该模型

#### Scenario: 轻量导入 registry
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import CLS-token Transformer fusion 模型依赖

#### Scenario: 内置组件列表可发现
- **WHEN** 开发者按扩展文档触发默认模型模块导入后查看 `MODELS.list()`
- **THEN** 输出 MUST 包含 `cls_token_transformer_fusion`
- **AND** 输出 MUST 继续包含现有 canonical fusion 模型注册名

### Requirement: CSI 组件注册
项目 MUST 通过现有组件注册表注册 CSI encoder 和可选 CSI 模型入口，使用户能通过配置构建 pilot dual-view CSI encoder，并复用现有 `modular_sequence` 训练流程。

#### Scenario: 按名称构建 CSI encoder
- **WHEN** 配置中指定 `type: pilot_dual_view_csi` 及其初始化参数
- **THEN** 系统 MUST 通过 `ENCODERS` 注册表返回 CSI encoder 实例
- **AND** 构建参数 MUST 支持 `output_dim`、`d_model`、pilot estimation、dual-view、tokenizer、temporal 和 dropout 相关字段

#### Scenario: 默认组件导入包含 CSI 模块
- **WHEN** 构建流程调用默认组件导入函数后再构建 `pilot_dual_view_csi`
- **THEN** `ENCODERS` 注册表 MUST 包含 `pilot_dual_view_csi`
- **AND** 注册表轻量导入边界 MUST 与现有 registry 语义一致

### Requirement: CSI 注册错误可诊断
CSI 相关注册错误 MUST 使用现有注册表错误风格，并在未知名称、重复名称或缺失必需参数时提供清晰错误信息。

#### Scenario: 请求未知 CSI encoder
- **WHEN** 配置中引用未注册的 CSI encoder 名称
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含请求的组件名称、注册表名称和可用组件列表

#### Scenario: CSI 构建参数非法
- **WHEN** 配置中引用 `pilot_dual_view_csi` 但提供非法 `view_fusion` 或非正数 `pilot_len`
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含非法字段或原始构建错误

### Requirement: Hist 组件注册已退役
组件注册表 MUST 不再注册 HiST-Beam/Hist 专用模型、loss、adapter、prototype 或 workflow 组件。旧注册名 MUST 被识别为已删除或未知名称，并给出当前支持范围。

#### Scenario: hist_beam_fusion 构建失败
- **WHEN** 用户请求构建 `hist_beam_fusion`
- **THEN** registry 或配置构建 MUST 拒绝该名称
- **AND** 错误信息 MUST 说明 Hist/HiST-Beam 研究线已退役

#### Scenario: Hist variants 不作为模型注册名
- **WHEN** 默认组件导入完成后开发者查看 `MODELS` 注册名
- **THEN** 注册名 MUST 不包含 HiST-Beam variants、P3/radio prototype variants、image-only Hist probe variants 或 history-anchor Hist variants
- **AND** 当前主线模型注册名 MUST 继续可用

### Requirement: 已删除组件错误包含 Hist 迁移方向
当用户引用 Hist 旧组件名且该名称仍由当前迁移 guard 覆盖时，错误信息 MUST 区分退役研究线与普通拼写错误，并指向当前推荐 workflow 或说明无兼容迁移。若本 change 删除对应 guard table，Hist 旧组件名 MAY 回落为普通未知名称，但仍 MUST 不注册为 current 可构建组件。

#### Scenario: Hist 旧模型名错误可诊断
- **WHEN** 用户配置 `model.primary.type: hist_beam_fusion`
- **THEN** 系统 MUST 拒绝构建并包含请求名称
- **AND** 若 Hist guard 被保留，错误信息 MUST 提示使用当前 supervised、adapter、GPS candidate、residual fusion 或其它保留 workflow；若 guard 已删除，系统 MAY 使用普通 unknown-name 错误

### Requirement: JEPA downstream pooler 和 adapter 注册
项目 MUST 通过轻量组件构建边界暴露 JEPA downstream pooler。内置 mean pooler 和 GPS-query attention pooler MUST 能通过配置名称构建；identity adapter MAY 作为默认 no-op 路径内联，而不是必须注册为独立 adapter。未知 pooler 名称 MUST 使用现有 registry 错误风格报告；未知 adapter 名称只有在非 identity adapter 配置面被保留时才需要注册表式错误。

#### Scenario: 按名称构建 mean pooler
- **WHEN** `jepa_context_image` 配置声明 downstream pooler 为 `mean`
- **THEN** 系统 MUST 构建 mean pooler
- **AND** 该 pooler MUST 接收 patch tokens `[B,T,N,D]` 并输出 `[B,T,D]`

#### Scenario: 按名称构建 GPS-query pooler
- **WHEN** `jepa_context_image` 配置声明 downstream pooler 为 `gps_query_attention`
- **THEN** 系统 MUST 构建 GPS-query attention pooler
- **AND** 构建参数 MUST 支持 `k_queries`、`num_heads`、`condition_dim`、`latent_dim`、dropout 和 condition source

#### Scenario: identity adapter 内联为 no-op
- **WHEN** `jepa_context_image` 配置未声明 adapter 或声明 adapter 为 `identity`
- **THEN** 系统 MUST 使用不改变输入 shape 的无操作路径
- **AND** 现有配置 MUST 无需新增 adapter 字段即可运行

#### Scenario: 未知 JEPA downstream 组件可诊断
- **WHEN** 用户配置不存在的 JEPA downstream pooler 名称
- **THEN** 系统 MUST 拒绝构建
- **AND** 错误信息 MUST 包含请求名称、组件类别和可用 pooler 名称

### Requirement: JEPA downstream 注册保持轻量导入
JEPA downstream pooler 的注册 MUST 不破坏 registry 轻量导入边界。导入 `kd_sensing.registries` MUST 不 eager import torch model implementation、dataset、diagnostics、训练器或 checkpoint 文件；默认组件导入流程 MUST 显式注册内置 JEPA downstream pooler。identity adapter 若内联为 no-op，则不需要默认注册流程。

#### Scenario: 轻量导入 registry 不触发 JEPA model
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import `kd_sensing.models.jepa` 或 JEPA downstream 实现模块

#### Scenario: 默认组件导入后可构建 JEPA downstream pooler
- **WHEN** 构建流程调用默认组件导入函数
- **THEN** 内置 JEPA downstream pooler MUST 完成注册
- **AND** 用户配置中的内置 pooler 名称 MUST 可解析

### Requirement: Difficulty operator 注册表
项目 MUST 提供 difficulty operator 注册边界，用于按字符串名称注册、查询和构建 GPS、image 和未来模态输入难度 operator。该注册边界 MAY 复用现有 `Registry` 实现或新增窄 registry，但 MUST 保持轻量导入，不得在导入 registry 时 eager import dataset、model、diagnostics renderer、training loop 或大型视觉依赖。

#### Scenario: 按名称构建 GPS delay operator
- **WHEN** 配置指定 difficulty operator `gps_temporal_delay` 及其参数
- **THEN** 系统 MUST 通过 difficulty operator registry 构建该 operator
- **AND** 训练、评估和 benchmark MUST 能复用同一注册名

#### Scenario: 轻量导入 difficulty registry
- **WHEN** 开发者执行 `import kd_sensing.registries` 或导入 difficulty registry 窄模块
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入默认 dataset、model、diagnostics renderer、torchvision 权重接口或训练循环

#### Scenario: 未知 difficulty operator 错误可诊断
- **WHEN** 配置引用未注册 difficulty operator
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含 registry 名称、请求 operator 和可用 operator 列表

### Requirement: 默认 difficulty operator 显式注册
内置 difficulty operators MUST 通过显式默认组件导入或 difficulty 专用默认注册函数完成注册。构建流程在解析或应用 difficulty profile 前 MUST 触发该注册动作；仅导入 registry 对象 MUST 不自动注册所有重依赖 operator。

#### Scenario: 构建前导入默认 difficulty operators
- **WHEN** 配置加载或 benchmark runner 需要解析内置 GPS/image difficulty profile
- **THEN** 构建流程 MUST 先触发默认 difficulty operator 注册
- **AND** registry MUST 包含 GPS noise、GPS async、image degradation 等内置注册名

#### Scenario: 自定义 difficulty operator 可插拔
- **WHEN** 开发者在自定义模块中注册新的 image difficulty operator 并在配置中引用
- **THEN** 系统 MUST 能在该模块被显式导入后解析并构建该 operator
- **AND** 训练和 benchmark 主循环 MUST 不需要为该 operator 增加专用分支

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

### Requirement: 默认组件导入登记新增模型组件
新增内置模型子组件或整模型例外 MUST 被默认组件导入流程显式登记，同时保持 `kd_sensing.registries` 轻量可导入。默认组件导入 MUST 不通过兼容 facade、仓库扫描或旧聚合模块发现组件。

#### Scenario: 新组件可通过默认导入发现
- **WHEN** 构建流程调用 `import_default_components()` 后查询对应 registry
- **THEN** 新增内置 encoder/projector/core/head 或例外模型注册名 MUST 出现在 registry 列表中
- **AND** 仅导入 `kd_sensing.registries` MUST 不 eager import dataset、trainer、torchvision 权重接口或 checkpoint 文件

### Requirement: 扩展文档区分默认和例外注册
组件发现和扩展文档 MUST 将新增 baseline 的默认路径描述为模块化配置或子组件注册。直接注册 `MODELS` 的示例 MUST 位于 whole-model exception 小节，并说明需要 OpenSpec 设计理由和 focused tests。

#### Scenario: 文档默认示例使用模块化组件
- **WHEN** 开发者阅读 Add a Model 或新增 baseline 指南
- **THEN** 首个示例 MUST 展示 `modular_sequence` 配置或子组件 registry
- **AND** 文档 MUST 不把直接 `@MODELS.register` 整模型作为普通 baseline 的默认建议

### Requirement: Legacy model registry names are retired with migration guards
项目 MUST 将已退役的 legacy model、encoder、core 和 head 注册名排除在 current 可构建组件之外。对仍有当前迁移价值的旧名称，项目 MAY 保留 removed guard 并给出明确迁移目标；对完全退役且不再承诺兼容的旧名称，项目 MAY 删除 guard table 并让 registry 使用普通 unknown-name 错误。

#### Scenario: 旧整模型注册名被拒绝
- **WHEN** 用户通过 `MODELS.build()` 请求 `radar_strong`、`gps_lightweight`、`mmwave_strong`、`fusion_lightweight` 或其它本 change 退役的旧整模型注册名
- **THEN** 系统 MUST 拒绝构建该名称
- **AND** 若 removed guard 被保留，错误信息 MUST 包含请求名称、registry 名称和 `modular_sequence` 迁移目标；若 guard 已删除，系统 MAY 使用普通 unknown-name 错误

#### Scenario: 旧别名被拒绝
- **WHEN** 用户请求 `modular_sequence_model`、`gps_only_neural_baseline`、`jepa_token_transformer` 或 `safe_residual_reranker`
- **THEN** 系统 MUST 不把这些名称注册为 current 可构建组件
- **AND** 若别名仍在 current migration table 中，错误信息 MUST 指向对应 canonical 名称或配置路径

#### Scenario: feature extractor 不作为完整模型列出
- **WHEN** 默认组件导入完成后开发者查看 `MODELS.list()`
- **THEN** 输出 MUST NOT 包含 `radar_feature_extractor`、`lidar_feature_extractor` 或 `mmwave_feature_extractor`
- **AND** 对应 feature extractor 类 MAY 继续通过窄模块导入或由 encoder 组件内部复用

#### Scenario: current registry discovery 只列当前入口
- **WHEN** 文档、架构摘要或架构边界测试检查 current registry surface
- **THEN** current model/encoder/core/head 清单 MUST 不把 removed guard 名称展示为可推荐入口
- **AND** removed 名称 MAY 出现在退役边界、migration table 或普通错误路径中

