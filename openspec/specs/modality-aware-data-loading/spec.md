# modality-aware-data-loading Specification

## Purpose
Define how enabled modalities are resolved from configuration, how dataset sample fields are selected, and how modality-specific normalization/cache behavior remains compatible across training and evaluation.
## Requirements
### Requirement: Scenario 9 按模态选择加载样本
Scenario 9 dataset MUST 根据训练或评估配置中的启用模态加载样本字段。未启用模态的文件 MUST 不被读取，未启用模态的输入字段 MUST 不出现在样本字典中，且未启用模态的路径列或文件缺失不得阻止当前任务运行。dataset MUST 始终加载 beam 历史标签和 future beam 目标标签。

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
- **THEN** dataset MUST 只读取 RGB image、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 radar、GPS、LiDAR、mmWave 或 image motion cache 逻辑
- **AND** 返回样本 MUST 包含 `image`

#### Scenario: fusion 按 modalities 读取输入
- **WHEN** 用户运行 `experiment.task: fusion` 且配置 `modalities: ["radar", "mmwave"]`
- **THEN** dataset MUST 只读取 radar、mmWave、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、GPS 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: 启用模态推导
数据构建流程 MUST 从 `experiment.task`、fusion teacher/student `modalities` 和显式 dataset 开关推导有序启用模态，并将该选择传递给 dataset、训练 batch 准备和评估 batch 准备。默认 fusion 模态 MUST 保持既有 `["image", "radar"]` 行为。

#### Scenario: 单模态任务推导
- **WHEN** 配置的 `experiment.task` 是 `image`、`radar`、`gps`、`lidar` 或 `mmwave`
- **THEN** 数据构建流程 MUST 将启用模态推导为对应单模态
- **AND** 显式启用的 GPS、LiDAR 或 mmWave dataset 开关 MUST 与任务模态保持一致或被清晰拒绝

#### Scenario: fusion teacher/student 模态一致
- **WHEN** fusion KD 配置同时定义 teacher 和 student `modalities`
- **THEN** 数据构建流程 MUST 使用 teacher 与 student 的并集作为 dataset 启用模态
- **AND** 如果 teacher 与 student 模态不一致且配置未声明受支持跨模态蒸馏，系统 MUST 抛出清晰错误

#### Scenario: 未配置 fusion modalities
- **WHEN** fusion 配置没有显式设置 teacher 或 student `modalities`
- **THEN** 数据构建流程 MUST 使用 `["image", "radar"]`
- **AND** dataset MUST 保持旧 image+radar fusion 的样本字段兼容

#### Scenario: mmWave dataset 开关冲突
- **WHEN** 配置设置 `data.dataset.use_mmwave: true` 但 `experiment.task` 或 fusion `modalities` 未启用 `mmwave`
- **THEN** 系统 MUST 拒绝构建 dataset
- **AND** 错误信息 MUST 指出 `use_mmwave` 与启用模态冲突

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

### Requirement: DeepSense6G 场景感知数据构建
数据构建流程 MUST 根据 DeepSense6G 场景选择解析 dataset 类型、数据根目录和 split CSV。现有 `scenario9` 配置 MUST 继续可构建，并 MUST 等价于选择 Scenario 9。

#### Scenario: scenario9 兼容旧配置
- **WHEN** 用户运行包含 `data.dataset.type: scenario9` 的旧配置
- **THEN** 数据构建流程 MUST 将该配置视为 DeepSense6G Scenario 9
- **AND** dataset 返回字段、启用模态推导和标签张量 shape MUST 保持兼容

#### Scenario: 通用 deepsense6g 类型选择 Scenario 32
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`
- **THEN** 数据构建流程 MUST 构建 Scenario 32 对应的 DeepSense6G dataset
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: split metadata 记录场景
- **WHEN** 训练或评估构建 train/test dataset
- **THEN** split metadata MUST 记录每个 split 的 `scene_id`、`scene_slug`、CSV 路径和样本数
- **AND** 这些字段 MUST 出现在最终配置、运行日志或测试报告中

#### Scenario: 场景不影响模态按需读取
- **WHEN** 用户在 Scenario 32 上运行 mmWave-only 或 GPS+mmWave fusion 配置
- **THEN** dataset MUST 只读取启用模态所需文件和 beam label 文件
- **AND** 未启用模态的缺失文件不得阻止该任务运行

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

### Requirement: 序列 CSV 使用 balanced_seq split 协议
DeepSense6G 序列 CSV 生成流程 MUST 使用单一的 `balanced_seq` train/test split 协议。split MUST 以完整 `seq_index` 为最小单位，MUST 不把同一 `seq_index` 的滑动窗口同时分配到 train 和 test，且 MUST 保持每个窗口仍只在单个 `seq_index` 内生成。

#### Scenario: split 可复现
- **WHEN** 用户使用相同原始 CSV、`training_set_pct`、`split_seed` 和 seq 数量控制配置
- **THEN** 序列生成流程 MUST 在重复运行时产生相同的 train/test `seq_index` 集合
- **AND** 系统 MUST 允许不同 `split_seed` 产生不同集合

#### Scenario: 标签分布感知选择 test seq
- **WHEN** 用户运行序列 CSV 预处理
- **THEN** 序列生成流程 MUST 基于生成后的窗口标签统计选择完整 test seq
- **AND** test 窗口数量 MUST 尽量接近配置的目标测试比例或显式 test seq 数
- **AND** test label 分布 MUST 尽量接近全量窗口 label 分布

#### Scenario: 小 seq 数场景的最少验证 seq
- **WHEN** 用户配置 `min_test_sequences` 且可用 `seq_index` 数量足以满足该约束
- **THEN** 序列生成流程 MUST 至少选择该数量的 test seq
- **AND** 如果该约束与显式 `test_sequence_count` 冲突，系统 MUST 抛出清晰错误或按文档定义的优先级处理

### Requirement: 序列 split metadata 可追踪
序列 CSV 预处理 MUST 为生成的 train/test split 记录可机器读取的 metadata。metadata MUST 足以解释当前 split 的策略、seed、seq 分配、窗口数和主要 label 分布。

#### Scenario: 写出 split metadata
- **WHEN** 用户运行序列 CSV 预处理
- **THEN** 系统 MUST 写出 split metadata sidecar
- **AND** metadata MUST 包含 `split_protocol: balanced_seq`、`split_seed`、`training_set_pct`、train/test `seq_index` 列表、train/test 窗口数和输出 CSV 路径

#### Scenario: 记录标签分布摘要
- **WHEN** 序列 CSV 中包含 beam 标签路径
- **THEN** split metadata MUST 记录 train/test 的 label 分布摘要
- **AND** 摘要 MUST 至少覆盖当前时隙标签或所有训练目标时隙中的一种明确口径

#### Scenario: 新统一 split 必须有 metadata
- **WHEN** 用户使用新预处理配置生成默认统一 split CSV
- **THEN** train/test CSV 旁 MUST 存在 split metadata sidecar
- **AND** metadata 中的 train/test 窗口数 MUST 与输出 CSV 行数一致

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

