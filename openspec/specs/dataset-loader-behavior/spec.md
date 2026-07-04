# dataset-loader-behavior Specification

## Purpose
定义通用 DataLoader、样本窗口、portion 采样、标签 shape、按需加载和轻量缓存行为，承接跨数据集但不属于 descriptor/schema 本身的运行时加载契约。
## Requirements
### Requirement: Scenario 9 按模态选择加载样本
DeepSense6G dataset MUST 根据训练或评估配置中的启用模态加载样本字段。未启用模态的文件 MUST 不被读取，未启用模态的输入字段 MUST 不出现在样本字典中，且未启用模态的路径列或文件缺失不得阻止当前任务运行。dataset MUST 始终加载 beam 历史标签和 future beam 目标标签。Scenario 9 MUST 通过 `data.dataset.type: deepsense6g` 和 `data.dataset.scene: 9` 选择，不得通过 `scene-specific dataset class alias` 或 `the scene-9 dataset-type spelling` 选择。

#### Scenario: GPS-only 不读取 image 或 radar 文件
- **WHEN** 用户运行 `experiment.task: gps` 的训练或评估配置
- **THEN** dataset MUST 只读取 GPS、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image 或 radar map 加载逻辑
- **AND** 返回样本 MUST 不包含 `image`、`radar_ra` 或 `radar_da`

#### Scenario: LiDAR-only 不读取 image 或 radar 文件
- **WHEN** 用户运行 `experiment.task: lidar` 的训练或评估配置
- **THEN** dataset MUST 只读取 LiDAR、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、radar map 或 GPS 加载逻辑
- **AND** 返回样本 MUST 不包含 `image`、`radar_ra`、`radar_da` 或 `gps`

#### Scenario: mmWave-only 不读取其它输入模态文件
- **WHEN** 用户运行 `experiment.task: mmwave` 的训练或评估配置
- **THEN** dataset MUST 只读取 mmWave、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、radar map、GPS 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 包含 `mmwave`
- **AND** 返回样本 MUST 不包含 `image`、`radar_ra`、`radar_da`、`gps` 或 `lidar`

#### Scenario: radar-only 只读取 radar 输入
- **WHEN** 用户运行 `experiment.task: radar` 的训练或评估配置
- **THEN** dataset MUST 只读取 radar、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、GPS、LiDAR 或 mmWave 加载逻辑
- **AND** 返回样本 MUST 包含 `radar_ra` 和 `radar_da`

#### Scenario: image-only 只读取 image 输入
- **WHEN** 用户运行 `experiment.task: image` 的训练或评估配置
- **THEN** dataset MUST 只读取 image、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 radar、GPS、LiDAR 或 mmWave 加载逻辑
- **AND** 返回样本 MUST 包含 `image`

#### Scenario: fusion 按 modalities 读取输入
- **WHEN** 用户运行 `experiment.task: fusion` 且配置 `modalities: ["radar", "mmwave"]`
- **THEN** dataset MUST 只读取 radar、mmWave、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、GPS 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: 标签张量维度稳定
Scenario 9 dataset MUST 返回稳定维度的 `input_beam` 和 `target_beam`。单样本 `target_beam` MUST 保持形状 `[num_pred]`，batch 后 MUST 保持形状 `[batch_size, num_pred]`，包括 `num_pred=1` 的情况。`prepare_labels()` MUST 仅使用 `target_beam[:, :num_pred]` 生成训练标签，不得将 `input_beam` 的最后一个历史 beam 拼入 label。

#### Scenario: num_pred 为 1
- **WHEN** dataset 配置 `num_pred: 1` 且读取一个样本
- **THEN** 返回样本的 `target_beam` MUST 是一维张量且长度为 1
- **AND** DataLoader batch 的 `target_beam` MUST 是二维张量且第二维长度为 1
- **AND** `prepare_labels` MUST 返回形状 `[batch_size, 1]` 的未来标签
- **AND** `prepare_labels` MUST 不包含 `input_beam[-1]`

#### Scenario: num_pred 大于 1
- **WHEN** dataset 配置 `num_pred: 3` 且读取一个 batch
- **THEN** batch 的 `target_beam` MUST 保持 `[batch_size, 3]`
- **AND** `prepare_labels` MUST 返回 `[t+1, t+2, t+3]` 对应的 `[batch_size, 3]` 标签
- **AND** 训练、验证和评估指标 MUST 按 `num_pred` 个未来时隙计算
- **AND** 标签 MUST 不包含历史窗口最后一个 beam

### Requirement: DataLoader 运行参数可配置
训练和评估入口 MUST 支持通过配置控制 DataLoader 运行参数，包括 batch size、`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor` 和 `drop_last`。当 `num_workers=0` 时，系统 MUST 不传入仅适用于多 worker 的参数。

#### Scenario: 多 worker DataLoader
- **WHEN** 配置设置 `data.dataloader.num_workers: 4`、`persistent_workers: true`、`pin_memory: true` 和 `prefetch_factor: 2`
- **THEN** 训练和评估 DataLoader MUST 使用这些参数
- **AND** train loader MUST 按配置决定是否 `drop_last`

#### Scenario: 单进程 DataLoader
- **WHEN** 配置设置 `data.dataloader.num_workers: 0`
- **THEN** 系统 MUST 不向 DataLoader 传入 `persistent_workers` 或 `prefetch_factor`
- **AND** DataLoader MUST 能在 CPU-only smoke test 中正常迭代

#### Scenario: 评估复用 loader 参数解析
- **WHEN** 用户运行评估入口并配置 DataLoader 参数
- **THEN** 评估入口 MUST 使用与训练入口一致的参数解析逻辑
- **AND** 评估入口 MUST 保持 `shuffle: false`

### Requirement: Scenario 9 序列窗口完整生成
Scenario 9 序列 CSV 生成流程 MUST 包含每个 `seq_index` 内所有合法的滑动窗口。对长度为 `N` 的单个 `seq_index`，输入长度为 `in_len`、预测长度为 `out_len` 时，合法窗口数 MUST 为 `max(N - in_len - out_len + 1, 0)`。

#### Scenario: 包含最后一个合法窗口
- **WHEN** 某个 `seq_index` 包含 `N` 行，且 `N == in_len + out_len`
- **THEN** 序列生成 MUST 为该 `seq_index` 产生 1 个窗口
- **AND** 输出窗口 MUST 使用前 `in_len` 行作为历史输入，后 `out_len` 行作为未来目标

#### Scenario: 多个 seq_index 分别计算窗口
- **WHEN** 原始 CSV 包含多个 `seq_index`
- **THEN** 系统 MUST 在每个 `seq_index` 内独立生成窗口
- **AND** 系统 MUST 不跨 `seq_index` 拼接历史输入或未来目标

### Requirement: 小比例 portion 采样代表性
Dataset 样本构建流程 MUST 明确 `portion` 小比例采样语义。默认 `portion < 1.0` 时，系统 MUST 使用确定性、可复现且覆盖 CSV 全局分布的采样策略，不得默认只取 CSV 头部连续样本。采样策略、seed 和最终样本数 MUST 可记录到运行 metadata。

#### Scenario: portion 不取连续头部样本
- **WHEN** 用户设置 `portion: 0.05` 且 CSV 样本数大于 20
- **THEN** 默认采样结果 MUST 不等价于 `head(int(len * portion))`
- **AND** 采样结果 MUST 使用稳定 seed 或确定性索引，保证重复运行样本集合一致

#### Scenario: portion 保留 seq_index 覆盖
- **WHEN** 序列 CSV 包含 `seq_index` 列且 `portion < 1.0`
- **THEN** 采样策略 MUST 尽可能覆盖完整 `seq_index` 范围
- **AND** 运行 metadata MUST 记录采样后的样本数和涉及的 `seq_index` 范围

#### Scenario: portion 全量采样
- **WHEN** 用户设置 `portion: 1.0`
- **THEN** Dataset MUST 使用 CSV 中全部样本
- **AND** 样本顺序 MUST 与 CSV 原始顺序保持兼容

### Requirement: Beam label 轻量缓存
Scenario 9 dataset MUST 支持在当前 split 内缓存 beam 文本解析结果。该缓存 MUST 是轻量整数映射，不得把 image、radar、GPS 或 LiDAR 大数组放入同一初始化缓存中。

#### Scenario: Dataset 初始化建立 beam label cache
- **WHEN** 配置启用 eager beam label cache
- **THEN** dataset MAY 扫描当前 split 唯一 input/future beam 路径并保存整数 label 映射
- **AND** 初始化缓存 MUST 只保存 path 和 int label，不保存大模态数组

#### Scenario: Dataset 按需建立 beam label cache
- **WHEN** 配置使用 lazy beam label cache
- **THEN** dataset MUST 在第一次遇到 beam path 时解析并缓存 label
- **AND** 后续遇到同一路径 MUST 复用缓存 label

### Requirement: 多任务目标按需加载
DeepSense6G dataset MUST 在配置启用多任务辅助监督时按需生成并返回遮挡和位置目标。未启用多任务辅助监督时，dataset MUST 不读取 future GPS target，不拟合遮挡阈值，且不得改变现有按模态懒加载行为。

#### Scenario: beam-only 不读取辅助目标
- **WHEN** 用户运行未启用多任务辅助监督的 image、radar、GPS、LiDAR、mmWave 或 fusion 配置
- **THEN** dataset MUST 不返回 `occlusion_label`、`position_target`、`occlusion_valid` 或 `position_valid`
- **AND** dataset MUST 不读取 future GPS target 列
- **AND** dataset MUST 不扫描 beam power 文件拟合遮挡阈值

#### Scenario: 启用遮挡目标
- **WHEN** dataset 配置启用 `occlusion_target.enabled: true`
- **THEN** dataset MUST 使用 `future_beam` 路径对应的 64-beam power vector 生成每个预测时隙的 `occlusion_label`
- **AND** 返回样本 MUST 包含形状 `[num_pred]` 的 `occlusion_label` 和 `occlusion_valid`

#### Scenario: 启用位置目标
- **WHEN** dataset 配置启用 `position_target.enabled: true`
- **THEN** dataset MUST 返回形状 `[num_pred, 2]` 的 `position_target`
- **AND** 返回样本 MUST 包含形状 `[num_pred]` 的 `position_valid`

### Requirement: 辅助目标 artifact 复用
数据构建流程 MUST 将训练 split 拟合出的遮挡阈值和位置目标归一化统计作为运行 artifact 记录，并在测试 split 和独立评估中复用。测试 split MUST 不重新拟合这些统计量。

#### Scenario: 训练保存辅助目标统计
- **WHEN** 训练 dataset 启用遮挡目标或位置目标归一化
- **THEN** 训练流程 MUST 保存对应统计 artifact 到运行目录
- **AND** final config 或 run metadata MUST 记录 artifact 路径和关键统计值

#### Scenario: 测试复用训练统计
- **WHEN** 构建 test dataset 且辅助目标需要训练统计
- **THEN** 数据构建流程 MUST 将训练 dataset 的统计对象传给 test dataset
- **AND** test dataset MUST 不扫描测试集拟合阈值或 scaler

#### Scenario: 独立评估加载 artifact
- **WHEN** 用户运行评估入口并加载启用了多任务辅助监督的 checkpoint
- **THEN** 评估流程 MUST 从 checkpoint registry、normalization artifact 或显式配置加载遮挡阈值和位置统计
- **AND** 如果缺失必要 artifact，系统 MUST 抛出清晰错误

### Requirement: DeepSense6G target provider
DeepSense6G dataset MUST 通过 target provider 组件构造 beam、occlusion、position 和 multitask 所需 target 字段。dataset 主类 MAY 协调 provider 的生命周期，但新增 target 类型 MUST 不要求修改主类的核心 `__getitem__` 取样流程。

#### Scenario: occlusion target 由 provider 生成
- **WHEN** dataset 配置启用 `occlusion_target`
- **THEN** occlusion target provider MUST 读取所需 mmWave power 或统计信息并生成 `occlusion_label` 与 `occlusion_valid`
- **AND** 返回样本字段、dtype 和 shape MUST 与现有 occlusion objective 训练兼容

#### Scenario: position target 由 provider 生成
- **WHEN** dataset 配置启用 `position_target`
- **THEN** position target provider MUST 读取或构造位置 target 并生成 `position_target` 与 `position_valid`
- **AND** provider MUST 复用既有 normalization/scaler 语义

#### Scenario: 未启用 target 不读取相关资源
- **WHEN** 当前 objective 不需要 occlusion 或 position target
- **THEN** 对应 target provider MUST 不读取 mmWave power、GPS future position 或 position scaler 资源
- **AND** 返回样本 MUST 不包含未启用 target 字段

### Requirement: DeepSense6G 模态 loader 组件
DeepSense6G dataset MUST 将 image、radar、GPS、LiDAR 和 mmWave 的文件读取、cache 访问和特征构造委托给模态 loader 组件。未启用模态的 loader MUST 不初始化重资源，也不得读取该模态文件。

#### Scenario: GPS+mmWave fusion 只初始化相关 loader
- **WHEN** 配置启用 fusion modalities `["gps", "mmwave"]`
- **THEN** dataset MUST 只初始化 GPS、mmWave、beam label 和启用 target 所需组件
- **AND** dataset MUST 不初始化 image、radar 或 LiDAR loader 的重资源

#### Scenario: 新增模态 loader 不影响 target provider
- **WHEN** 开发者新增或修改某个输入模态 loader
- **THEN** 变更 MUST 不要求编辑 occlusion 或 position target provider
- **AND** target provider 测试 MUST 继续通过

### Requirement: DeepSense6G CSV 相对路径基准
DeepSense6G dataset MUST 继续以解析后的 scene root 作为 CSV 内相对文件路径的基准目录。将场景目录移动到 `dataset/DeepSense6G/scenario*` 后，CSV 内现有相对路径格式 MUST 不需要增加 `DeepSense6G` 前缀。

#### Scenario: 读取新规范目录下的相对路径
- **WHEN** dataset 的 `data_root` 为 `dataset/DeepSense6G/scenario31` 且 CSV 内某个 radar 路径为 `/unit1/radar_data_RA/sample.npy`
- **THEN** 文件读取 MUST 解析到 `dataset/DeepSense6G/scenario31/unit1/radar_data_RA/sample.npy`
- **AND** 系统 MUST 不把该路径解析到 `dataset/unit1/radar_data_RA/sample.npy`

#### Scenario: 读取显式旧目录下的相对路径
- **WHEN** dataset 的 `data_root` 被显式设置为 `dataset/scenario31` 且 CSV 内某个 mmWave 路径为 `/unit1/pwr/sample.txt`
- **THEN** 文件读取 MUST 解析到 `dataset/scenario31/unit1/pwr/sample.txt`
- **AND** 系统 MUST 不要求用户修改 CSV 内相对路径

### Requirement: HDF5/cache-backed 懒加载
HDF5 或 cache-backed dataset 如果属于当前保留数据集，MUST 在初始化时只读取 index、shape 和 metadata，不得物化全量 image、LiDAR、CSI/channel 大数组。未启用模态 MUST 完全跳过对应数据读取。Multimodal-NF HDF5/cache-backed lazy loading 不再作为支持路径。

#### Scenario: 保留 dataset 初始化不物化大数组
- **WHEN** 用户构建当前保留的 HDF5/cache-backed train dataset
- **THEN** dataset 初始化 MUST 不把全量大数组读入内存
- **AND** dataset MUST 只保留轻量 index、文件路径、key 和必要 metadata

#### Scenario: Multimodal-NF lazy loading 删除
- **WHEN** 用户运行 Multimodal-NF CSI-only、image-only 或 fusion 配置
- **THEN** dataset 构建 MUST 失败
- **AND** 系统 MUST 不进入 Multimodal-NF image、LiDAR 或 CSI lazy loading 分支

### Requirement: split metadata 和 artifact 复用
数据构建流程 MUST 记录并复用当前保留数据集的 split metadata 和必要 normalizer artifact。train split 拟合出的 artifact MUST 能传递给 val/test split，而不要求重新扫描全量数据。系统 MUST 不再复用 Multimodal-NF split metadata 或 codebook metadata。

#### Scenario: 保留 artifact 复用
- **WHEN** train dataset 已解析当前保留数据集所需的 normalizer 或 metadata
- **THEN** val/test dataset MUST 使用同一 metadata 或 artifact
- **AND** 如果 val/test metadata 与 train 不一致，系统 MUST 抛出清晰错误

#### Scenario: Multimodal-NF metadata 删除
- **WHEN** 训练或评估构建 dataloaders
- **THEN** run metadata MUST 不要求包含 Multimodal-NF split protocol、city 列表、input profiles、target schema 或 codebook metadata

### Requirement: Dataset hotspot 拆分必须保持 loader 行为
Dataset 重构 MUST 保持 lazy loading、enabled modality resolution、sample cache behavior、scaler fitting、no-future-leak target construction 和 run metadata。

#### Scenario: 拆分后样本契约兼容
- **WHEN** DeepSense6GDataset or MMWDataset helper boundaries change
- **THEN** existing sample keys, target tensors, auxiliary target metadata, cache metadata and warning behavior MUST remain compatible
- **AND** focused tests MUST 不要求真实数据文件

