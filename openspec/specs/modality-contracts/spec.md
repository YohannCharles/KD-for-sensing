# modality-contracts Specification

## Purpose
定义中心化模态顺序、dataset flag、batch key 和默认字段推导契约，确保配置、数据集、模型和诊断共享同一模态语义。
## Requirements
### Requirement: 中心化模态契约
项目 MUST 提供单一来源的模态契约，用于描述所有受支持模态的规范名称、固定顺序、dataset flag、样本字段、fusion 输入字段、默认 dataset/model 字段，以及是否支持 cache 或归一化 artifact。image modality MUST 不暴露 image motion cache、motion profile 或 motion encoder 推荐。

#### Scenario: 枚举受支持模态
- **WHEN** 开发者查询模态契约
- **THEN** 系统 MUST 返回固定顺序的 `image`、`radar`、`gps`、`lidar`、`mmwave` 和 `csi`
- **AND** 该顺序 MUST 被 canonical config、fusion 模态解析、dataset 构建和诊断配置复用

#### Scenario: 查询 image 模态元数据
- **WHEN** 开发者查询 `image` 模态契约
- **THEN** 系统 MUST 返回 image 对应的样本字段 `image`
- **AND** 系统 MUST 返回 fusion 输入字段 `image_batch`
- **AND** 系统 MUST 返回 RGB/ImageNet 输入契约
- **AND** 系统 MUST 不返回 image motion cache 能力

#### Scenario: 查询 radar 模态元数据
- **WHEN** 开发者查询 `radar` 模态契约
- **THEN** 系统 MUST 返回 radar 对应的样本字段 `radar_ra` 和 `radar_da`
- **AND** 系统 MUST 返回 fusion 输入字段 `radar_batch`

#### Scenario: 查询 CSI 模态元数据
- **WHEN** 开发者查询 `csi` 模态契约
- **THEN** 系统 MUST 返回 CSI 对应的样本字段 `csi`
- **AND** 系统 MUST 返回 fusion 输入字段 `csi_batch`
- **AND** 系统 MUST 返回 dataset flag `use_csi`
- **AND** 系统 MUST 返回 CSI RMS normalizer artifact key

### Requirement: 模态列表标准化
系统 MUST 通过模态契约标准化用户配置中的模态列表。标准化 MUST 拒绝未知模态、空列表和重复模态，并 MUST 按固定模态顺序返回结果。

#### Scenario: 标准化乱序 fusion 模态
- **WHEN** 用户配置 fusion `modalities: ["csi", "lidar", "image", "gps"]`
- **THEN** 系统 MUST 将有效模态标准化为 `["image", "gps", "lidar", "csi"]`
- **AND** dataset、batch 准备和模型构建 MUST 使用同一个标准化结果

#### Scenario: 拒绝未知模态
- **WHEN** 用户配置 `modalities: ["image", "thermal"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含未知模态和可用模态列表

#### Scenario: 拒绝重复模态
- **WHEN** 用户配置 `modalities: ["csi", "csi"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出模态列表包含重复项

### Requirement: 模态契约驱动 dataset flag
数据构建流程 MUST 通过模态契约生成 dataset flag 和相关默认字段，避免在多个模块中手写 `use_gps`、`use_lidar`、`use_mmwave`、`use_csi` 等分支。

#### Scenario: GPS、mmWave 与 CSI fusion 设置 dataset flag
- **WHEN** fusion 启用模态为 `["gps", "mmwave", "csi"]`
- **THEN** 数据构建流程 MUST 设置 `use_gps: true`、`use_mmwave: true` 和 `use_csi: true`
- **AND** 数据构建流程 MUST 不设置与未启用 LiDAR 相关的启用 flag

#### Scenario: 单模态 image 不启用可选模态 flag
- **WHEN** `experiment.task: image`
- **THEN** 数据构建流程 MUST 只启用 image 所需字段
- **AND** GPS、LiDAR、mmWave 和 CSI 的 dataset flag MUST 保持关闭或缺省关闭

### Requirement: 模态契约驱动 batch 输入
训练、验证、评估和诊断路径 MUST 使用模态契约确定 batch 字段到模型输入参数的映射。新增或调整模态输入键时，系统 MUST 不要求在训练、验证和评估循环中复制分支逻辑。

#### Scenario: fusion batch 输入映射
- **WHEN** fusion 启用模态为 `["radar", "mmwave", "csi"]`
- **THEN** batch 准备流程 MUST 从样本字段构建 `radar_batch`、`mmwave_batch` 和 `csi_batch`
- **AND** 模型调用 MUST 不传入未启用 image、GPS 或 LiDAR 的输入张量

#### Scenario: 单模态 batch 输入映射
- **WHEN** `experiment.task: csi`
- **THEN** batch 准备流程 MUST 构建 CSI 模型所需输入
- **AND** 训练和评估循环 MUST 通过统一映射调用模型

### Requirement: 模态契约文档化
项目文档 MUST 说明模态契约的职责，以及新增模态时必须补充的 dataset 字段、batch 字段、model 输入、cache/normalization 能力和测试。

#### Scenario: 按文档新增模态
- **WHEN** 开发者按照扩展文档新增一个实验性模态
- **THEN** 文档 MUST 指引开发者先新增模态契约
- **AND** 文档 MUST 指出后续需要补充 dataset 读取、模型注册、batch 准备和诊断显示逻辑

### Requirement: RGB image profile 元数据
模态契约 MUST 为 image modality 暴露 RGB/ImageNet 输入 profile 元数据。元数据 MUST 至少包含 profile 名称、期望通道数、默认空间尺寸、dataset 样本字段、fusion 输入字段、是否支持 cache、以及推荐 encoder 类型。

#### Scenario: 查询 RGB ImageNet profile 元数据
- **WHEN** 开发者查询 image modality 的 `rgb_imagenet` profile
- **THEN** 系统 MUST 返回通道数 3、默认空间尺寸 224x224、样本字段 `image`、fusion 输入字段 `image_batch`
- **AND** 系统 MUST 标记该 profile 不支持 image cache
- **AND** 系统 MUST 推荐 `resnet18_imagenet_rgb` encoder

### Requirement: Image profile 标准化
系统 MUST 通过模态契约或等价中心化函数标准化 `image_profile` 配置。标准化 MUST 拒绝未知或已删除 profile，并 MUST 为未配置 profile 的默认路径返回 `rgb_imagenet`。

#### Scenario: 默认配置标准化
- **WHEN** 用户配置启用 image modality 且未设置 `image_profile`
- **THEN** 标准化结果 MUST 为 `rgb_imagenet`
- **AND** dataset、batch 准备和模型构建 MUST 使用同一个标准化结果

#### Scenario: RGB 配置标准化
- **WHEN** 用户配置 `image_profile: rgb_imagenet`
- **THEN** 标准化结果 MUST 保留为 `rgb_imagenet`
- **AND** 后续配置校验 MUST 能据此要求 3 通道 RGB encoder

### Requirement: Batch 输入准备使用 RGB image profile
训练、验证、评估和诊断路径 MUST 使用标准化后的 image profile 决定 image batch 准备逻辑。batch 准备 MUST 在进入模型前形成明确的 `[B, T, 3, H, W]` tensor，并 MUST 使用统一的历史长度和 future padding 策略。

#### Scenario: RGB batch 准备
- **WHEN** image profile 为 `rgb_imagenet`
- **THEN** batch 准备 MUST 接受 dataset 返回的 RGB 帧序列
- **AND** 传给模型的通道数 MUST 为 3
- **AND** future padding MUST 不改变历史 RGB 帧的标准化值

### Requirement: Image 模态仅支持 RGB/ImageNet 输入
系统 MUST 将 image modality 的输入契约固定为 RGB/ImageNet 路径。配置解析、模态契约和模型构建 MUST 拒绝 `motion_mask` profile、motion cache 能力和 motion image encoder。

#### Scenario: 默认 image profile 为 RGB/ImageNet
- **WHEN** 开发者查询 image modality 的输入契约
- **THEN** 系统 MUST 返回 RGB/ImageNet 输入语义
- **AND** 系统 MUST 返回 3 通道、224x224 的默认空间尺寸
- **AND** 系统 MUST 标记该 image 输入不使用 image motion cache

#### Scenario: motion profile 不可解析
- **WHEN** 用户配置 `image_profile: motion_mask`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含 `motion_mask` 已删除和可用 image 输入契约

### Requirement: 模态 profile 契约
中心化模态契约 MUST 支持当前保留 dataset-specific input profile，用于在不新增模态名称的情况下表达同一模态在不同数据集中的输入语义、shape、默认字段和 batch 准备规则。profile 标准化 MUST 拒绝未知 profile，并 MUST 保持未配置旧 profile 时的既有行为。Multimodal-NF 专属 `uav_xyz_snapshot`、`xl_mimo_nf` 和 `point_cloud_xyz_10000` profiles MUST 不再作为支持 profile 保留，Raymobtime s008 专属 profile MUST 不再作为当前保留 profile。

#### Scenario: 保留数据集 profile 可查询
- **WHEN** 开发者查询当前保留数据集的 image、GPS、LiDAR、mmWave 或 CSI profile
- **THEN** 系统 MUST 返回对应 sample key、fusion input key 和输入语义
- **AND** 查询 MUST 不要求 Multimodal-NF 或 Raymobtime s008 profile 存在

#### Scenario: Multimodal-NF profile 被拒绝
- **WHEN** 用户配置 `uav_xyz_snapshot`、`xl_mimo_nf` 或 `point_cloud_xyz_10000`
- **THEN** 系统 MUST 拒绝该 profile 或因 dataset type 已退役而失败
- **AND** 错误信息 MUST 包含 profile 名称和当前可用 profile 列表

#### Scenario: Raymobtime profile 被拒绝
- **WHEN** 用户配置 Raymobtime s008 专属 coord、ray 或 LiDAR occupancy profile
- **THEN** 系统 MUST 拒绝该 profile 或因 Raymobtime s008 已退役而失败
- **AND** 错误信息 MUST 包含 Raymobtime s008 已退役或当前可用 profile 列表

### Requirement: profile 列表标准化
系统 MUST 能基于当前保留 dataset descriptor 和用户配置标准化启用模态对应的 input profiles。标准化 MUST 在 metadata 中记录每个模态的 resolved profile。系统 MUST 不再为 `data.dataset.type: multimodal_nf` 解析默认 profile。

#### Scenario: 保留 dataset 默认 profile
- **WHEN** 用户配置当前保留 dataset 并启用多个模态
- **THEN** 系统 MUST 解析这些模态在该 dataset 下的默认或显式 profile
- **AND** metadata MUST 记录 resolved profile

#### Scenario: Multimodal-NF 默认 profile 删除
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf`
- **THEN** 系统 MUST 不解析 image/lidar/gps/csi 的 Multimodal-NF 默认 profile
- **AND** 系统 MUST 报告该 dataset type 已退役

### Requirement: profile 驱动 batch 输入准备
训练、验证、评估和诊断路径 MUST 使用标准化后的当前保留 input profile 决定 batch shape 校验和必要转换。新增 profile 时，系统 MUST 不要求在每个训练循环复制 dataset-specific 分支。Multimodal-NF CSI 和 LiDAR 点云 batch 准备不再作为支持路径。

#### Scenario: 保留 profile batch 输入
- **WHEN** batch 包含当前保留 profile 的模态字段
- **THEN** runtime MUST 构造对应 input batch
- **AND** shape 校验和缺失字段错误 MUST 使用该 profile 的语义

#### Scenario: Multimodal-NF batch 输入删除
- **WHEN** batch 或配置请求 Multimodal-NF `xl_mimo_nf` CSI batch 或 `point_cloud_xyz_10000` LiDAR batch
- **THEN** runtime MUST 不构造这些 batch 输入
- **AND** 系统 MUST 报告 profile 或 dataset type 不受支持

### Requirement: Difficulty profile 复用 canonical modality keys
Difficulty profile MUST 使用中心化模态契约中的 canonical modality name、sample key 和 fusion input key 来声明 affected modality。难度 profile MUST 不新增 `gps_noisy`、`delayed_gps`、`image_hard` 等伪模态名称，也 MUST 不要求训练、评估或模型 forward 为每种难度新增专用输入分支。

#### Scenario: GPS difficulty 使用 gps canonical key
- **WHEN** profile 声明 GPS jitter、delay 或 dropout
- **THEN** affected modality MUST 标准化为 `gps`
- **AND** transform MUST 作用于当前 batch 的 GPS sample key，并由现有 `gps_batch` 准备路径消费

#### Scenario: 拒绝伪模态名称
- **WHEN** 用户在 modalities 或 difficulty affected modality 中配置 `delayed_gps`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 指向 canonical modality `gps` 和 difficulty profile 配置

### Requirement: Difficulty mask 与 metadata 字段语义
模态契约或等价中心化 helper MUST 定义 difficulty 产生的输入相关 mask/metadata 字段语义。GPS async/missing 字段至少 MUST 覆盖 valid、stale、delay steps、source index 和 dropout mask；image degradation 字段至少 MUST 覆盖 degradation type、severity、seed、frame range 和 optional mask。字段命名 MUST 避免与 target schema、auxiliary target 或 sensitive supervision 字段混淆。

#### Scenario: GPS async metadata 可查询
- **WHEN** 开发者查询 GPS modality 的 difficulty metadata fields
- **THEN** 系统 MUST 返回 `gps_valid_mask`、`gps_stale_mask`、`gps_delay_steps`、`gps_source_index` 和 `gps_dropout_mask` 或等价字段说明
- **AND** 这些字段 MUST 被标记为输入 reliability metadata，而不是 target supervision

#### Scenario: image degradation metadata 不改变 profile
- **WHEN** image difficulty operator 输出 degradation metadata
- **THEN** metadata MUST 记录为 difficulty metadata
- **AND** image input profile MUST 仍保持 `rgb_imagenet` 或配置解析后的当前 profile

### Requirement: Image observability metadata 字段
模态契约或等价中心化 helper MUST 定义通用 image observability difficulty metadata 字段语义。字段至少 MUST 覆盖 `image_valid_mask`、`image_observability_score`、`image_dropout_mask`、`image_burst_dropout_mask`、`image_degradation_metadata`、corruption type、severity、seed 和 frame range；字段 MUST 不依赖已退役 Scenario-D condition id。

#### Scenario: 查询 image observability metadata
- **WHEN** 开发者查询 image modality 的 difficulty metadata fields
- **THEN** 系统 MUST 返回 image valid mask、observability score、dropout/burst masks 和 degradation metadata 的字段说明
- **AND** 这些字段 MUST 被标记为输入 reliability metadata，而不是 target supervision 或辅助标签

#### Scenario: metadata 字段不创建伪模态
- **WHEN** 配置启用通用 image degradation 或 missing difficulty
- **THEN** affected modality MUST 仍标准化为 canonical `image`
- **AND** 系统 MUST 拒绝 `image_hard`、`missing_image_modality` 或其它伪模态名称

### Requirement: Reliability metadata 进入 batch 输入映射
训练和评估 batch 输入映射 MUST 能将通用 image/GPS reliability metadata 传递给显式支持的 current 模型，同时保持不支持该 metadata 的模型兼容。metadata 传递 MUST 不要求每个 difficulty condition 新增专用模型输入分支，也 MUST 不保留 Scenario-D/GPS-query benchmark 专属条件映射。

#### Scenario: observability-aware 模型接收 metadata
- **WHEN** current 模型配置声明需要 observability-aware fusion
- **THEN** batch 准备 MUST 向模型 forward 提供其声明的 image observability 和 GPS reliability metadata
- **AND** 缺少字段时 MUST 抛出清晰错误或记录配置声明的 fallback warning

#### Scenario: 普通 baseline 忽略 metadata
- **WHEN** standard Image ResNet+GPS 或其它 baseline 不声明 reliability metadata 输入
- **THEN** batch 准备 MUST 允许其忽略通用 reliability metadata
- **AND** run comparability metadata MUST 记录该模型是否消费 reliability metadata

### Requirement: 模态数据转换职责拆分
数据转换模块 MUST 按 image、radar、lidar、gps、mmwave 和通用 IO/cache/normalization 职责组织。新增或修改某个模态的数据读取、特征构造或 cache key 时，变更 MUST 不要求编辑其它模态的转换实现。项目 MUST 不再保留 `the transform facade module` 或 `the transform aggregate module` 作为兼容聚合入口。

#### Scenario: 修改 GPS 特征不触碰 LiDAR 转换
- **WHEN** 开发者修改 GPS feature sequence 构造
- **THEN** 变更 MUST 限定在 GPS 转换相关模块和测试
- **AND** 不需要修改 LiDAR BEV、image 或 mmWave feature 转换实现

#### Scenario: 旧 transforms import 被拒绝
- **WHEN** 现有代码从 `the transform facade module` 或 `the transform aggregate module` 导入转换函数或 scaler
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向 `kd_sensing.data.transform_ops.<modality>` 或通用 transform 子模块

### Requirement: 模态转换实现不得集中在 legacy 聚合模块
数据转换模块 MUST 将仍在使用的 image RGB、GPS、LiDAR、mmWave、radar、IO、cache 和 normalization 实现放入对应模块。`the transform aggregate module` MUST 不再存在或不再作为运行时入口导出任何符号。

#### Scenario: 修改 image 实现不触碰 LiDAR 实现
- **WHEN** 开发者修改 RGB image 加载或标准化逻辑
- **THEN** 主要变更 MUST 限定在 image 转换相关模块和测试
- **AND** 不需要编辑 LiDAR、GPS、mmWave 或 radar 转换实现

#### Scenario: 修改 GPS scaler 不触碰 image 实现
- **WHEN** 开发者修改 GPS feature 或 scaler 加载保存逻辑
- **THEN** 主要变更 MUST 限定在 GPS 或通用 normalization 模块
- **AND** 不需要编辑 image、LiDAR BEV 或 radar map 转换实现

#### Scenario: legacy 聚合模块引用被拒绝
- **WHEN** 开发者运行内部引用扫描
- **THEN** 扫描 MUST 拒绝 `the transform aggregate module`
- **AND** 扫描 MUST 指向对应的窄 transform 模块作为迁移路径

### Requirement: 启用模态解析唯一来源
训练、验证、评估、诊断和 dataset 构建路径 MUST 使用 `engine.modality_resolution` 或其公开 helper 解析启用模态。入口层不得新增 `_uses_gps`、`_uses_lidar`、`_uses_mmwave` 等重复配置推导 helper。

#### Scenario: evaluator 复用 modality resolution
- **WHEN** 评估入口需要判断当前配置是否启用 LiDAR 或 mmWave
- **THEN** 入口 MUST 调用统一模态解析 helper
- **AND** 不得在 evaluator 中维护独立的配置字段判断逻辑

#### Scenario: fusion teacher/student 模态冲突错误一致
- **WHEN** fusion 配置中 teacher 和 student modalities 不一致且未声明支持跨模态蒸馏
- **THEN** 训练和评估路径 MUST 抛出一致的错误信息
- **AND** 错误 MUST 来自统一模态解析逻辑

### Requirement: 启用模态推导
数据构建流程 MUST 从 `experiment.task`、fusion teacher/student `modalities` 和显式 dataset 开关推导有序启用模态，并将该选择传递给 dataset、训练 batch 准备和评估 batch 准备。默认 fusion 模态 MUST 保持既有 `["image", "radar"]` 行为。

#### Scenario: 单模态任务推导
- **WHEN** 配置的 `experiment.task` 是 `image`、`radar`、`gps`、`lidar`、`mmwave` 或 `csi`
- **THEN** 数据构建流程 MUST 将启用模态推导为对应单模态
- **AND** 显式启用的 GPS、LiDAR、mmWave 或 CSI dataset 开关 MUST 与任务模态保持一致或被清晰拒绝

#### Scenario: fusion teacher/student 模态一致
- **WHEN** fusion KD 配置同时定义 teacher 和 student `modalities`
- **THEN** 数据构建流程 MUST 使用 teacher 与 student 的并集作为 dataset 启用模态
- **AND** 如果 teacher 与 student 模态不一致且配置未声明受支持跨模态蒸馏，系统 MUST 抛出清晰错误

#### Scenario: 未配置 fusion modalities
- **WHEN** fusion 配置没有显式设置 teacher 或 student `modalities`
- **THEN** 数据构建流程 MUST 使用 `["image", "radar"]`
- **AND** dataset MUST 保持旧 image+radar fusion 的样本字段兼容

#### Scenario: CSI dataset 开关冲突
- **WHEN** 配置设置 `data.dataset.use_csi: true` 但 `experiment.task` 或 fusion `modalities` 未启用 `csi`
- **THEN** 系统 MUST 拒绝构建 dataset
- **AND** 错误信息 MUST 指出 `use_csi` 与启用模态冲突

### Requirement: RGB image 感知加载
Scenario 9 dataset 在启用 image modality 时 MUST 直接加载 RGB/ImageNet image 输入。系统 MUST 不再支持 motion mask 在线生成、motion mask cache 懒加载或 `motion_mask` profile。

#### Scenario: image-only 使用 RGB 输入
- **WHEN** 用户运行 image-only 配置
- **THEN** dataset MUST 读取当前样本所需的 RGB image 帧
- **AND** 返回样本 MUST 包含 `image` 和 label 字段
- **AND** 返回的 `image` MUST 可被 RGB/ImageNet image encoder 消费
- **AND** dataset MUST 不调用 motion mask 生成或 image motion cache 路径解析逻辑

#### Scenario: motion profile 被拒绝
- **WHEN** 用户配置 `image_profile: motion_mask`
- **THEN** dataset 或配置解析 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明 image motion 路径已删除且需要使用 RGB/ImageNet image 输入

### Requirement: Objective-aware 数据目标加载
数据加载流程 MUST 根据 `experiment.objective` 和模型需求启用对应 targets。未被当前 objective 或显式辅助配置使用的 targets MUST 不被强制读取或拟合 artifact。

#### Scenario: beam objective 不读取辅助目标
- **WHEN** `experiment.objective` 为 `beam` 且未显式启用辅助监督
- **THEN** dataset MUST 不要求 `occlusion_target` 或 `position_target`
- **AND** dataset MUST 不拟合遮挡阈值或位置 target scaler

#### Scenario: occlusion objective 启用遮挡目标
- **WHEN** `experiment.objective` 为 `occlusion`
- **THEN** dataset MUST 返回 `occlusion_label` 和 `occlusion_valid`
- **AND** dataset MUST 拟合或复用训练 split 的遮挡阈值 artifact

#### Scenario: position objective 启用位置目标
- **WHEN** `experiment.objective` 为 `position`
- **THEN** dataset MUST 返回 `position_target` 和 `position_valid`
- **AND** dataset MUST 拟合或复用训练 split 的位置 target scaler artifact，除非配置禁用 target normalization

#### Scenario: multitask objective 启用全部目标
- **WHEN** `experiment.objective` 为 `multitask`
- **THEN** dataset MUST 返回 beam、occlusion 和 position 目标所需的所有字段
- **AND** dataset MUST 保存和复用遮挡阈值与位置 target scaler artifacts

### Requirement: Objective-aware batch 准备
batch/runtime helper MUST 能把当前 objective 所需 targets 搬到目标 device，并保持无效位置 mask 与预测 horizon 对齐。

#### Scenario: occlusion batch targets
- **WHEN** batch 包含遮挡标签且 objective 需要遮挡目标
- **THEN** runtime MUST 返回 device 上的 `occlusion_label` 和 `occlusion_valid`
- **AND** 返回张量 MUST 裁剪或校验到 `num_pred` horizon

#### Scenario: position batch targets
- **WHEN** batch 包含位置目标且 objective 需要位置目标
- **THEN** runtime MUST 返回 device 上的 `position_target` 和 `position_valid`
- **AND** 返回张量 MUST 裁剪或校验到 `num_pred` horizon

#### Scenario: 缺失目标字段
- **WHEN** 当前 objective 需要某个 target 但 batch 缺少对应字段
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指明缺失字段和当前 `experiment.objective`

### Requirement: 自动 cache policy 下的模态感知 cache 访问
Scenario 9 dataset MUST 在自动 cache policy 下保持按模态访问数据。启用 LiDAR 时允许使用 LiDAR BEV cache；未启用 LiDAR 时 MUST 完全跳过 LiDAR cache 访问。启用 image 时 MUST 不访问 image motion cache。

#### Scenario: image-only 不使用 image motion cache
- **WHEN** 用户运行 image-only 配置且 `data.cache.policy: auto`
- **THEN** dataset MUST 使用 RGB/ImageNet image 输入
- **AND** dataset MUST 不读取、不创建、不写入 image motion cache
- **AND** 返回样本字段、shape 和 dtype MUST 与 RGB/ImageNet image 契约一致

#### Scenario: LiDAR fusion 自动使用 LiDAR cache
- **WHEN** 用户运行包含 LiDAR 的 fusion 配置且 `data.cache.policy: auto`
- **THEN** dataset MUST 对 LiDAR BEV 启用 cache 读取
- **AND** cache miss 时 dataset MUST 生成并写入缺失的 LiDAR BEV cache
- **AND** 返回样本字段、shape 和 dtype MUST 与未启用 cache 时一致

#### Scenario: 非相关模态不触发 cache 初始化
- **WHEN** 用户运行不包含 LiDAR 的单模态或 fusion 配置
- **THEN** dataset 初始化 MUST 不创建 LiDAR cache 目录
- **AND** dataset 取样 MUST 不调用 LiDAR cache path 解析逻辑
- **AND** dataset MUST 不调用任何 image motion cache path 解析逻辑

### Requirement: Fusion 模态选择配置
Fusion teacher 和 fusion student MUST 支持通过 `modalities` 配置选择参与融合的模态。`modalities` MUST 是 `image`、`radar`、`gps`、`lidar`、`mmwave`、`csi` 的非空列表；默认值 MUST 保持既有 image+radar 行为。

#### Scenario: 默认 fusion 模态
- **WHEN** 用户构建 fusion 模型且未显式配置 `modalities`
- **THEN** 系统 MUST 使用 `["image", "radar"]`
- **AND** 系统 MUST 保持旧 image+radar 配置的模型输入和输出行为兼容

#### Scenario: 配置全部模态
- **WHEN** 用户配置 `modalities: ["image", "radar", "gps", "lidar", "mmwave", "csi"]`
- **THEN** fusion 模型 MUST 创建 image、radar、gps、lidar、mmWave 和 CSI 六个分支
- **AND** fusion projection 的输入维度 MUST 与六个分支输出拼接维度一致

#### Scenario: 配置任意双模态
- **WHEN** 用户配置 `modalities` 为 `["image", "csi"]`、`["radar", "csi"]`、`["mmwave", "csi"]` 或其它合法双模态组合
- **THEN** fusion 模型 MUST 只创建被启用模态的分支
- **AND** forward MUST 只要求被启用模态对应的输入张量

#### Scenario: 配置单模态 fusion
- **WHEN** 用户配置 `modalities` 为 `["image"]`、`["radar"]`、`["gps"]`、`["lidar"]`、`["mmwave"]` 或 `["csi"]`
- **THEN** fusion 模型 MUST 能构建并运行
- **AND** fusion projection MUST 只接收该单模态分支输出

### Requirement: Fusion 模态配置校验
系统 MUST 对 fusion `modalities` 做显式校验。空列表、重复模态或未知模态 MUST 在模型构建时抛出清晰错误。

#### Scenario: 空模态列表
- **WHEN** 用户配置 `modalities: []`
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 指出至少需要一个模态

#### Scenario: 未知模态
- **WHEN** 用户配置 `modalities` 包含 `image`、`radar`、`gps`、`lidar`、`mmwave`、`csi` 之外的名称
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 包含非法模态名称

#### Scenario: 重复模态
- **WHEN** 用户配置 `modalities` 包含重复项
- **THEN** 系统 MUST 拒绝构建 fusion 模型
- **AND** 错误信息 MUST 指出模态不能重复

### Requirement: Fusion 输入准备遵循模态选择
训练、验证和评估流程在 `experiment.task: fusion` 下 MUST 根据配置的 `modalities` 准备输入。未启用的模态 MUST 不被要求存在于 batch 中。

#### Scenario: fusion 只启用 image 和 gps
- **WHEN** fusion 配置的 `modalities` 为 `["image", "gps"]`
- **THEN** batch 准备 MUST 构造 image 和 gps 输入
- **AND** batch 准备 MUST 不要求 `radar_ra`、`radar_da`、`lidar` 或 `mmwave`

#### Scenario: fusion 启用全部模态
- **WHEN** fusion 配置的 `modalities` 为 `["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** batch 准备 MUST 构造 image、radar、gps、lidar 和 mmWave 输入
- **AND** 五个输入的 batch 和 sequence 维度 MUST 对齐

#### Scenario: fusion 只启用 LiDAR
- **WHEN** fusion 配置的 `modalities` 为 `["lidar"]`
- **THEN** batch 准备 MUST 构造 LiDAR 输入
- **AND** batch 准备 MUST 不要求 image、radar、gps 或 mmWave 字段

#### Scenario: fusion 只启用 mmWave
- **WHEN** fusion 配置的 `modalities` 为 `["mmwave"]`
- **THEN** batch 准备 MUST 构造 mmWave 输入
- **AND** batch 准备 MUST 不要求 image、radar、GPS 或 LiDAR 字段

### Requirement: Fusion canonical 数据字段
canonical fusion 配置 MUST 根据 `modalities` 启用对应 dataset 字段，并不得要求未启用模态的数据列。启用 GPS 的配置 MUST 使用 GPS-Rel-Polar；启用 LiDAR 的配置 MUST 使用 LiDAR BEV 默认字段，并 MUST 沿用 LiDAR 懒加载和内存有界归一化语义；启用 mmWave 的配置 MUST 使用 64 维 dB receive-power 特征，并 MUST 复用训练集 mmWave scaler。

#### Scenario: 启用 GPS 的 fusion 配置
- **WHEN** canonical fusion 配置的 `modalities` 包含 `gps`
- **THEN** 配置 MUST 设置 `data.dataset.use_gps: true`
- **AND** 配置 MUST 设置 `gps_feature_mode: relative_polar`
- **AND** teacher 和 student 的 `gps_input_size` MUST 为 3

#### Scenario: 启用 LiDAR 的 fusion 配置
- **WHEN** canonical fusion 配置的 `modalities` 包含 `lidar`
- **THEN** 配置 MUST 设置 `data.dataset.use_lidar: true`
- **AND** 配置 MUST 提供 LiDAR BEV size、ROI 和归一化默认字段
- **AND** LiDAR 归一化默认字段 MUST 不要求 dataset 初始化阶段全量读取训练 split
- **AND** teacher 和 student MUST 使用与 LiDAR BEV 输入通道一致的 `lidar_channels`

#### Scenario: 启用 mmWave 的 fusion 配置
- **WHEN** canonical fusion 配置的 `modalities` 包含 `mmwave`
- **THEN** 配置 MUST 设置 `data.dataset.use_mmwave: true`
- **AND** 配置 MUST 设置 `mmwave_normalize: true`
- **AND** teacher 和 student 的 `mmwave_input_size` MUST 为 64

#### Scenario: fusion LiDAR streaming stats 显式启用
- **WHEN** canonical fusion 配置的 `modalities` 包含 `lidar` 且用户显式启用 LiDAR streaming stats
- **THEN** fusion dataloader MUST 使用与 LiDAR-only 配置相同的流式 stats 计算或 stats 文件复用逻辑
- **AND** 系统 MUST 不为 fusion 入口恢复全量 BEV concatenate 行为

#### Scenario: fusion mmWave scaler 复用
- **WHEN** canonical fusion 配置的 `modalities` 包含 `mmwave`
- **THEN** fusion dataloader MUST 使用与 mmWave-only 配置相同的训练集 scaler fit、保存和测试集复用逻辑
- **AND** 系统 MUST 不在测试 split 上重新 fit mmWave scaler

#### Scenario: 未启用模态不强制要求数据字段
- **WHEN** canonical fusion 配置的 `modalities` 不包含某个模态
- **THEN** 训练、验证和评估的 batch 准备 MUST 不要求该模态对应输入存在
- **AND** 模型 forward MUST 只接收启用模态对应的张量

### Requirement: Modular fusion 复用现有模态选择语义
新的模块化 fusion 入口 MUST 复用现有 `modalities` 校验、固定模态顺序和 batch 输入字段语义。未启用的模态 MUST 不被 dataset、batch 准备、encoder 或 core 要求存在。

#### Scenario: 模块化 fusion 只启用 image 和 gps
- **WHEN** 模块化 fusion 配置的 `modalities` 为 `["image", "gps"]`
- **THEN** batch 准备 MUST 只构造 `image_batch` 和 `gps_batch`
- **AND** 模型 forward MUST 不要求 radar、LiDAR 或 mmWave 输入

#### Scenario: 模块化 fusion 启用全部模态
- **WHEN** 模块化 fusion 配置的 `modalities` 为 `["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** 系统 MUST 为五个模态构建 encoder 和 projector
- **AND** representation core 接收的模态顺序 MUST 遵循模态契约固定顺序

### Requirement: 四模态 missing pattern 标准构造
系统 MUST 提供统一的四模态 missing pattern 构造 helper，用于训练评估、复评、BTAPA 分析和 summary。标准输出名称 MUST 使用 `gps`、`image`、`radar`、`lidar`，标准四模态顺序 MUST 为 `["gps", "image", "radar", "lidar"]`，并 MUST 支持常见大小写和旧字段 alias 映射到 canonical 名称。

#### Scenario: 查询标准 pattern mask
- **WHEN** 调用方以标准四模态顺序请求 `radar_only`
- **THEN** 系统 MUST 返回 `[0, 0, 1, 0]`
- **AND** `full`、`missing_gps`、`missing_image`、`missing_radar`、`missing_lidar`、`gps_only`、`image_only`、`radar_only` 和 `lidar_only` MUST 使用同一套构造逻辑

#### Scenario: 保留 missing_gps 与 non_gps_only 两个名称
- **WHEN** 当前四模态设置同时请求 `missing_gps` 和 `non_gps_only`
- **THEN** 两者 MAY 映射到相同 mask `[0, 1, 1, 1]`
- **AND** 输出表 MUST 保留两个 pattern 名称，避免旧结果读取和横向比较失败

#### Scenario: avg_missing 不是直接 mask
- **WHEN** 调用方请求 `avg_missing`
- **THEN** 系统 MUST 将其识别为聚合 pattern
- **AND** 系统 MUST 不把 `avg_missing` 当作可直接 forward 的单个 modality mask

#### Scenario: alias 标准化
- **WHEN** 调用方传入 `GPS`、`RGB`、`rad` 或大小写不同的模态名称
- **THEN** 系统 MUST 标准化为 `gps`、`image`、`radar` 或 `lidar`
- **AND** 未知或重复模态 MUST 抛出清晰错误

### Requirement: missing pattern 分类 helper
系统 MUST 提供统一 missing pattern API 覆盖标准 mask、pattern name 反查、标准 pattern 列表和 pattern 分类。标准四模态顺序 MUST 为 `["gps", "image", "radar", "lidar"]`。

#### Scenario: weak single modality 分类
- **WHEN** 调用 `is_weak_single_modality_pattern("radar_only")` 或 `is_weak_single_modality_pattern("lidar_only")`
- **THEN** 系统 MUST 返回 true
- **AND** 对 `gps_only`、`image_only`、`missing_gps` MUST 返回 false

#### Scenario: sensing only 分类
- **WHEN** 调用 `is_sensing_only_pattern` 判断 `image_only`、`radar_only`、`lidar_only`、`missing_gps` 或 `non_gps_only`
- **THEN** 系统 MUST 返回 true
- **AND** 对 `gps_only` MUST 返回 false

#### Scenario: 标准 pattern 列表包含聚合项
- **WHEN** 调用 `list_standard_missing_patterns(include_avg=True)`
- **THEN** 返回列表 MUST 包含 `avg_missing`
- **AND** `avg_missing` MUST 被识别为聚合项而不是可直接 forward 的 mask

