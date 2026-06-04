# modality-aware-data-loading Specification

## Purpose
Define how enabled modalities are resolved from configuration, how dataset sample fields are selected, and how modality-specific normalization/cache behavior remains compatible across training and evaluation.
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
数据构建流程 MUST 根据 DeepSense6G 场景选择和 dataset layout descriptor 解析数据根目录和 split CSV。canonical 配置 MUST 使用 `data.dataset.type: deepsense6g`。旧 `scenario9`、`scenario31` 和 `scenario32` dataset type MUST 不再可构建。

#### Scenario: 旧 scenario9 配置被拒绝
- **WHEN** 用户运行包含 `the scene-9 dataset-type spelling` 的旧配置
- **THEN** 数据构建流程 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明应使用 `data.dataset.type: deepsense6g` 和 `data.dataset.scene: 9`

#### Scenario: 通用 deepsense6g 类型默认选择 Scenario 31
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且未显式设置 `data.dataset.scene`
- **THEN** 数据构建流程 MUST 构建 Scenario 31 对应的 DeepSense6G dataset
- **AND** 数据根目录 MUST 默认为 `dataset/DeepSense6G/scenario31`
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: 显式选择 Scenario 32
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`
- **THEN** 数据构建流程 MUST 构建 Scenario 32 对应的 DeepSense6G dataset
- **AND** 数据根目录 MUST 默认为 `dataset/DeepSense6G/scenario32`
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: split metadata 记录场景
- **WHEN** 训练或评估构建 train/test dataset
- **THEN** split metadata MUST 记录每个 split 的 `scene_id`、`scene_slug`、CSV 路径和样本数
- **AND** 这些字段 MUST 出现在最终配置、运行日志或测试报告中

#### Scenario: 场景不影响模态按需读取
- **WHEN** 用户在任一受支持 DeepSense6G 场景上运行 mmWave-only 或 GPS+mmWave fusion 配置
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

### Requirement: 序列预处理输出 future 位置列
序列 CSV 预处理 MUST 支持显式输出 future GPS/BS GPS 位置目标列。该开关启用时，每个合法窗口 MUST 在保留现有 `beam` 和 `future_beam` 列的同时，输出与预测 horizon 对齐的 `future_gps` 和 `future_bs_gps` 列。

#### Scenario: 生成 future GPS target 列
- **WHEN** 用户运行序列预处理并设置 `include_position_targets: true`
- **THEN** 输出 CSV MUST 包含 `future_gps1..future_gpsN`
- **AND** 输出 CSV MUST 包含 `future_bs_gps1..future_bs_gpsN`
- **AND** `N` MUST 等于配置的预测长度 `out_len`

#### Scenario: 不启用时保持旧 CSV 结构
- **WHEN** 用户运行序列预处理但未启用 `include_position_targets`
- **THEN** 输出 CSV MUST 保持现有列结构兼容
- **AND** 旧的 beam-only dataset MUST 能继续读取该 CSV

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

### Requirement: Snapshot next-frame 序列预处理
序列 CSV 预处理 MUST 支持 snapshot next-frame 协议。该协议 MUST 在 Scenario 31 上使用 `in_len=1` 和 `out_len=1` 生成所有合法窗口，并保留当前帧输入与下一帧监督目标所需的列。

#### Scenario: 生成 snapshot 窗口
- **WHEN** 用户运行 snapshot next-frame 预处理配置
- **THEN** 预处理 MUST 设置 `in_len: 1` 和 `out_len: 1`
- **AND** 每个窗口 MUST 包含 `camera1`、`radar1`、`gps1`、`bs_gps1`、`lidar1`、`mmwave1`、`beam1`、`future_beam1` 和 `seq_index`
- **AND** 窗口 MUST 不跨 `seq_index` 拼接

#### Scenario: 生成 position target 列
- **WHEN** snapshot 预处理配置启用 `include_position_targets: true`
- **THEN** 输出 CSV MUST 包含 `future_gps1` 和 `future_bs_gps1`
- **AND** position objective MUST 使用这两个 future 列构造下一帧位置 target

### Requirement: Snapshot 80/20 sequence split
Snapshot next-frame 预处理 MUST 以完整 `seq_index` 为单位生成 80% train 和 20% validation split。split metadata MUST 明确记录这是 snapshot 协议，不得伪装成历史窗口统一 split。

#### Scenario: 写出 train/validation CSV
- **WHEN** snapshot next-frame 预处理完成
- **THEN** 系统 MUST 写出 `train_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** 系统 MUST 写出 `val_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** validation 窗口数量 MUST 尽量接近全量 snapshot 窗口的 20%

#### Scenario: split metadata 记录 snapshot 协议
- **WHEN** snapshot next-frame 预处理写出 split metadata
- **THEN** metadata MUST 包含 `split_protocol: snapshot_next_frame_balanced_seq`
- **AND** metadata MUST 包含 `training_set_pct: 0.8`
- **AND** metadata MUST 包含 train/validation `seq_index` 列表、窗口数、输出 CSV 路径和 label 分布摘要

#### Scenario: 不复用历史窗口 metadata
- **WHEN** snapshot 配置构建 dataset
- **THEN** 数据构建流程 MUST 加载 snapshot split metadata
- **AND** 如果 metadata 显示 `in_len` 或 `out_len` 不是 1，系统 MUST 拒绝该配置

### Requirement: Snapshot split-dependent artifact 隔离
依赖训练 split 拟合的 normalization、cache stats 或 target stats MUST 基于 snapshot train split 重新 fit。评估或验证 snapshot checkpoint 时，系统 MUST 使用同一 snapshot run 保存的 artifact 或与 snapshot split fingerprint 匹配的 artifact。

#### Scenario: mmWave scaler 使用 snapshot train split
- **WHEN** snapshot 配置启用 mmWave normalization
- **THEN** mmWave scaler MUST 只使用 `train_seqs_SNAPSHOT_NEXT_FRAME.csv` 中的 `mmwave1` fit
- **AND** validation dataset MUST 复用训练 split scaler
- **AND** 系统 MUST 不复用历史窗口 run 的 mmWave scaler

#### Scenario: LiDAR streaming stats 使用 snapshot train split
- **WHEN** snapshot 配置启用 LiDAR streaming stats
- **THEN** LiDAR normalizer MUST 只基于 snapshot train split 拟合统计量
- **AND** artifact metadata MUST 记录 snapshot split metadata path 和 fingerprint

#### Scenario: target stats 使用 snapshot train split
- **WHEN** snapshot objective 需要 occlusion threshold 或 position target scaler
- **THEN** threshold 或 scaler MUST 只从 snapshot train split 拟合
- **AND** validation split MUST 不参与拟合

#### Scenario: frame-level cache 可复用
- **WHEN** LiDAR BEV cache 或等价帧级 cache 只依赖原始文件路径和预处理参数
- **THEN** snapshot 配置 MAY 复用既有帧级 cache
- **AND** split-dependent normalizer/stat artifact MUST 仍与 snapshot split fingerprint 绑定

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

### Requirement: DeepSense6G 预处理路径重定向
DeepSense6G 序列 CSV 预处理 MUST 使用 dataset layout descriptor 解析 scene root。命令行或配置中的场景覆盖 MUST 同时更新 `preprocessing.data_root` 和默认 `preprocessing.csv_path` 到目标场景的规范目录，除非用户显式提供自定义绝对路径。

#### Scenario: 预处理默认 Scenario 31 路径
- **WHEN** 用户运行默认 DeepSense6G sequence CSV 预处理配置
- **THEN** `preprocessing.data_root` MUST 指向 `dataset/DeepSense6G/scenario31`
- **AND** `preprocessing.csv_path` MUST 指向 `dataset/DeepSense6G/scenario31/scenario31_RA.csv`

#### Scenario: 预处理场景覆盖到 Scenario 9
- **WHEN** 用户在 sequence CSV 预处理中覆盖 `data.dataset.scene: 9`
- **THEN** `preprocessing.data_root` MUST 更新为 `dataset/DeepSense6G/scenario9`
- **AND** 默认 `preprocessing.csv_path` MUST 更新为 `dataset/DeepSense6G/scenario9/scenario9_RA.csv`

### Requirement: MMW prepared manifests are loadable by modality-aware datasets
数据构建流程 MUST 能识别 MMW 准备流程生成的 manifest/CSV，并在配置选择 `data.dataset.type: mmw` 与 `data.dataset.scene: town10_skybridge_seed24` 时构建对应 dataset。启用模态推导、按需读取、beam 历史标签和 future beam 目标标签的语义 MUST 与现有 beam 预测流程保持一致。

#### Scenario: MMW mmWave-only 按需读取
- **WHEN** 用户使用 MMW manifest 运行 `experiment.task: mmwave`
- **THEN** dataset MUST 只读取历史 `mmwave*` power vector、`beam*` 和 `future_beam*` 标签文件
- **AND** dataset MUST 不读取 image、LiDAR、GPS 或 RSU radar 文件
- **AND** 返回样本 MUST 包含 `mmwave`、`input_beam` 和 `target_beam`

#### Scenario: MMW image+mmWave fusion 按需读取
- **WHEN** 用户使用 MMW manifest 运行 fusion 配置且启用 `["image", "mmwave"]`
- **THEN** dataset MUST 读取历史前向 RGB image、历史 mmWave power vector、历史 beam 和 future beam 标签
- **AND** dataset MUST 不要求未启用的 LiDAR、GPS 或 RSU radar 文件存在
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: MMW dataset returns stable beam and modality tensors
MMW dataset MUST 返回与现有训练流程兼容的 `input_beam` 和 `target_beam` 张量。启用 MMW 派生 mmWave 输入时，`mmwave` MUST 为 `[seq_len, 64]` 的 `torch.float32` 张量；启用 image、LiDAR 或 GPS 时，对应字段 MUST 使用现有 batch 准备流程可消费的稳定 shape 和 dtype。

#### Scenario: MMW beam 标签 shape 稳定
- **WHEN** MMW dataset 配置 `seq_len=8` 且 `num_pred=3`
- **THEN** 单样本 `input_beam` MUST 为长度 8 的整数张量
- **AND** 单样本 `target_beam` MUST 为长度 3 的整数张量
- **AND** batch 后 `target_beam` MUST 保持 `[batch_size, 3]`

#### Scenario: MMW mmWave 张量 shape 稳定
- **WHEN** MMW dataset 启用 mmWave modality
- **THEN** 单样本 `mmwave` MUST 为 `torch.float32`
- **AND** `mmwave` shape MUST 为 `[seq_len, 64]`
- **AND** 每个时隙 MUST 与同一行 CSV 的 `beam*` 历史标签时隙对齐

### Requirement: CSI 按模态选择加载样本
DeepSense6G/MMW dataset MUST 根据启用模态决定是否加载 CSI。未启用 CSI 时，CSI 路径列或文件缺失不得阻止当前任务运行；启用 CSI 时，dataset MUST 返回 `csi` 字段并保持其它未启用模态不读取。

#### Scenario: CSI-only 不读取其它输入模态文件
- **WHEN** 用户运行 `experiment.task: csi` 的训练或评估配置
- **THEN** dataset MUST 只读取 CSI、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、radar map、GPS、LiDAR 或 mmWave 加载逻辑
- **AND** 返回样本 MUST 包含 `csi`

#### Scenario: fusion 按 modalities 读取 CSI
- **WHEN** 用户运行 `experiment.task: fusion` 且配置 `modalities: ["gps", "csi"]`
- **THEN** dataset MUST 只读取 GPS、CSI、`input_beam` 和 `target_beam` 所需文件
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: CSI normalizer artifact 复用
数据构建流程 MUST 将训练集 CSI RMS normalizer 从 train dataset 传递给 test dataset，并允许训练/评估 metadata 记录该统计。

#### Scenario: dataloader 复用 CSI RMS
- **WHEN** `build_dataloaders` 构建启用 CSI 的 train 和 test dataset
- **THEN** train dataset MUST 先准备 CSI RMS normalizer
- **AND** test dataset MUST 接收同一个 CSI RMS normalizer 或等价数值

### Requirement: descriptor 驱动 dataset 构建
数据构建流程 MUST 根据当前保留 dataset descriptor 决定 split 解析、默认路径、storage kind、enabled modalities、input profiles 和 target schema。非 CSV 数据集 MUST 不被强制套用 DeepSense6G 的 train/test CSV 规则。已退役的 `multimodal_nf` descriptor、HDF5 index 和 cache index MUST 不再作为支持构建路径。

#### Scenario: Multimodal-NF dataset 构建失败
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf`
- **THEN** data factory MUST 拒绝该 dataset type
- **AND** 系统 MUST 不解析 `dataset/MultimodalNF`、HDF5 index 或 cache index

#### Scenario: 构建 DeepSense6G CSV dataset
- **WHEN** 用户配置 `data.dataset.type: deepsense6g`
- **THEN** data factory MUST 继续使用 DeepSense6G scene 和 split CSV 规则
- **AND** 现有 `train_csv_name`、`val_csv_name`、`test_csv_name` 覆盖行为 MUST 保持兼容

### Requirement: enabled modalities 与 profile 一起传递
数据构建流程 MUST 从实验任务、fusion 模态和当前保留 dataset descriptor 推导启用模态，并将标准化后的 input profiles 传递给 dataset、batch 准备和 run metadata。系统 MUST 不再解析或传递 Multimodal-NF 专属 profiles。

#### Scenario: 保留 fusion profile 传递
- **WHEN** 用户运行当前保留 fusion 配置并启用多个模态
- **THEN** data factory MUST 设置 `enabled_modalities`
- **AND** data factory MUST 传递每个保留模态的 resolved profile
- **AND** run metadata MUST 记录实际启用模态和 profile

#### Scenario: Multimodal-NF profile 拒绝
- **WHEN** 用户为退役 Multimodal-NF 配置 `image`、`lidar`、`gps` 或 `csi` profile
- **THEN** 系统 MUST 拒绝该 dataset type 或 profile
- **AND** 错误信息 MUST 指出 Multimodal-NF 已退役

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

### Requirement: MMW dataset 初始化内存有界
MMW dataset 初始化 MUST 避免为了 normalizer、CSV 派生列或 metadata 准备而无界持有所有样本的大数组。GPS、mmWave 和 CSI 等模态的 normalizer 拟合 MUST 使用 streaming 或可释放的临时统计，并 MUST 在拟合完成后避免把 per-sample sequence cache 常驻到 DataLoader worker。

#### Scenario: GPS/mmWave scaler 拟合不保留全量样本缓存
- **WHEN** MMW train dataset 启用 GPS 或 mmWave normalization
- **THEN** scaler 拟合 MUST 能通过 streaming 或临时数组完成
- **AND** 拟合完成后 dataset MUST 不保留所有样本的 GPS/mmWave sequence 大数组缓存
- **AND** runtime metadata MUST 记录 scaler 来源、样本数和是否使用 streaming 拟合

#### Scenario: DataLoader worker 不复制初始化大缓存
- **WHEN** MMW dataset 使用多 worker DataLoader
- **THEN** worker 进程 MUST 不因 dataset 初始化阶段的 per-sample feature cache 而复制全量样本大数组
- **AND** profile 或 metadata MUST 能报告 worker 内存风险相关配置

### Requirement: MMW image 序列按需加载与缓存等价
MMW image modality MUST 按 enabled modalities 和 seq_len 读取 RGB/ImageNet image 序列。启用 image-derived cache 时，dataset MUST 保持与原始 image 读取路径一致的样本字段、shape、dtype 和 label 语义。

#### Scenario: image-derived cache 保持 batch 契约
- **WHEN** MMW fusion 配置启用 image modality、`seq_len=8` 和 image-derived cache
- **THEN** 单样本 `image` tensor MUST 保持 `[seq_len, 3, H, W]`
- **AND** batch 后 image 输入 MUST 与未启用 cache 时的 shape 和 dtype 一致
- **AND** `input_beam`、`target_beam`、GPS 和 mmWave 字段 MUST 不因 image cache 改变

#### Scenario: 未启用 image 不读取 image 路径
- **WHEN** MMW fusion 配置的 modalities 为 `["gps", "mmwave"]`
- **THEN** dataset MUST 不读取 camera 列对应文件
- **AND** dataset MUST 不初始化 image transform 或 image-derived cache

### Requirement: LOSO stage dataset 构建边界
LOSO 数据构建流程 MUST 支持按 stage 构建当前阶段所需的数据集和 DataLoader，避免 source training 阶段提前构建 target adapt/test dataset。

#### Scenario: source_train 只构建 source loader
- **WHEN** LOSO executor 进入 `source_train` stage
- **THEN** 系统 MUST 只构建 source train dataset 和 loader
- **AND** 系统 MUST 不构建 target adapt 或 target test dataset

#### Scenario: target stage 延迟构建 target loader
- **WHEN** LOSO executor 进入 target adaptation 或 target test evaluation stage
- **THEN** 系统 MUST 在该 stage 内构建所需 target dataset 和 loader
- **AND** source stage 的 DataLoader worker MUST 已关闭或不再持有

### Requirement: Image-only probe batch 字段 allowlist
数据构建、collate 和 batch preparation MUST 在 image-only legal probe 中按启用模态和合法标签字段输出 batch。原始 manifest、CSV 或本地数据文件中存在 GPS、LiDAR、radar、mmWave、CSI、channel、path 或 beam_power 字段，不得导致这些字段进入模型输入、loss、target adaptation 或 evaluation payload。

#### Scenario: image-only batch 只暴露合法字段
- **WHEN** resolved modalities 等价于 `["image"]` 且 `protocol.image_only=true`
- **THEN** batch MUST 包含 image 输入
- **AND** batch MUST 包含 beam label 或现有 canonical target beam label
- **AND** batch MUST 在可用时包含 `scene`、`sample_id` 和 `split`
- **AND** batch MUST NOT 包含 `gps`、`lidar`、`radar`、`mmwave`、`csi`、`channel`、`path` 或 `beam_power` 作为可被模型、loss 或 adaptation 消费的字段

#### Scenario: collate 不要求禁用模态 key
- **WHEN** image-only dataset sample 不包含 GPS、LiDAR、radar、mmWave、CSI、channel、path 或 beam_power key
- **THEN** collate MUST 成功构造 batch
- **AND** dataloader one batch smoke test MUST 能在 `conda run -n kd_mm_beam` 环境中通过

#### Scenario: 原始字段存在不等于被消费
- **WHEN** 原始样本文件或 manifest 记录 path、radio、channel、beam_power、GPS 或 LiDAR 可用性
- **THEN** image-only batch preparation MUST 不把这些字段传给模型、loss、adaptation 或 evaluator
- **AND** run metadata MUST 区分 `available_fields` 与 `consumed_fields`

### Requirement: Image-only probe label 使用边界
image-only legal probe 的 dataset 和 batch preparation MUST 只把 beam label 作为 supervised target 暴露给 source training、target support adaptation 和 target_test evaluation。target adaptation MUST NOT 暴露 target test label、path/radio/channel label、beam_power argmax 或任何禁用 oracle 字段。

#### Scenario: target support 暴露 beam label
- **WHEN** 构建 target support dataloader 或 support feature cache
- **THEN** batch/cache MUST 暴露 support beam label
- **AND** support label source MUST 记录到 sampling metadata

#### Scenario: target test label 只用于 evaluation
- **WHEN** 构建 target test dataloader 或 target_test feature cache
- **THEN** target test beam label MUST 只在 evaluation scope 用于指标计算
- **AND** adaptation、threshold selection、temperature fitting、target prior 初始化和 prototype 构建 MUST NOT 读取 target test beam label

#### Scenario: 禁用 beam_power 离线指标时记录 unavailable
- **WHEN** image-only probe 禁用 `beam_power`
- **THEN** evaluation MUST NOT 为 BPL dB 或 NRP 读取 beam_power 作为输入或 adaptation 信号
- **AND** 如果无法合法计算 BPL dB 或 NRP，metrics MUST 将对应指标标记为 unavailable 并记录原因

### Requirement: Geometry-residual label 字段按需加载
数据加载流程 MUST 在 `label_space.type: geometry_residual` 时按需加载或派生 geometry-residual label 字段。未启用 geometry-residual label_space 时，dataset MUST 不要求 position/geometry 字段，也不得改变现有 absolute beam label sample keys。

#### Scenario: geometry_residual 启用时返回新增标签字段
- **WHEN** 用户运行启用 `label_space.type: geometry_residual` 的配置
- **THEN** dataset 或 target provider MUST 返回 absolute beam label、geometry coarse beam 和 residual label 可用字段
- **AND** batch preparation MUST 保持这些字段 shape 可默认 collate

#### Scenario: 默认 absolute 配置不读取 geometry
- **WHEN** 用户运行现有 absolute beam classifier 配置
- **THEN** dataset MUST 继续按启用模态读取 sensing input 和 absolute beam label
- **AND** 缺少 GPS/pose/relative geometry 不得阻止 dataset 构建

### Requirement: target-shot split 字段隔离
数据加载流程 MUST 根据 split artifact 中的 subset 标记构建 source、target_labeled、target_unlabeled 和 target_test dataloader。target_unlabeled loader MUST 能提供 sensing input 和非监督 metadata，但训练 payload MUST 不暴露可作为监督的 target labels。

#### Scenario: target_unlabeled loader 隔离监督字段
- **WHEN** 构建 target_unlabeled adaptation loader
- **THEN** batch metadata MUST 标记 subset 为 `target_unlabeled`
- **AND** training payload MUST 不允许 loss 访问 beam/residual supervision 字段

#### Scenario: target_test loader 只用于评估
- **WHEN** 构建 target_test loader
- **THEN** batch MAY 包含 evaluation metrics 所需 label
- **AND** run metadata MUST 标记 target_test labels 只可在 evaluation scope 使用

### Requirement: MMW calibrated hard label loading
MMW dataset MUST support returning calibrated hard beam labels when `data.dataset.beam_label_calibration.enabled=true`. Calibration MUST apply to historical `input_beam` and future `target_beam` while preserving existing tensor shapes and modality-aware loading behavior.

#### Scenario: calibrated input 和 target beam shape 稳定
- **WHEN** MMW dataset 配置 `seq_len=8`、`num_pred=3` 且启用 beam label calibration
- **THEN** 单样本 `input_beam` MUST 仍为长度 8 的整数张量
- **AND** 单样本 `target_beam` MUST 仍为长度 3 的整数张量
- **AND** 所有合法 label MUST 位于 `[0, num_classes)` 的 calibrated label space

#### Scenario: 显式 future_beam_label 字段被映射
- **WHEN** MMW split CSV 包含 `future_beam_label1` 或等价显式 raw label 字段
- **THEN** dataset MUST 在启用 calibration 时将该 raw label 映射为 calibrated `target_beam`
- **AND** metadata MUST preserve the original raw label value for audit

#### Scenario: beam label cache 区分 mapping
- **WHEN** beam label cache 为 eager 或 lazy 且 calibration 配置发生变化
- **THEN** dataset MUST NOT reuse cached calibrated labels from a different mapping fingerprint
- **AND** cache diagnostics MUST record the active mapping fingerprint

#### Scenario: 未启用模态仍不读取
- **WHEN** MMW fusion 配置启用 `["gps", "mmwave"]` 且启用 beam label calibration
- **THEN** dataset MUST only read GPS、mmWave、beam labels and enabled targets
- **AND** calibration MUST NOT cause image、LiDAR、radar、CSI、channel 或 path 文件被额外读取 as sensing inputs

### Requirement: DeepSense6G residual modality discovery
Residual manifest builder MUST automatically discover DeepSense6G optional modality resources and precomputed features without requiring every modality to exist.

#### Scenario: 自动发现预计算 feature
- **WHEN** manifest builder 扫描 DeepSense6G resources
- **THEN** 系统 MUST 优先识别 `.npy`、`.npz`、`.pt`、`.csv` 和 `.parquet` 形式的 precomputed feature
- **AND** 系统 MUST 将可用 feature path 写入 manifest 对应列

#### Scenario: 自动发现 sensor path
- **WHEN** image、LiDAR 或 radar path 在原始 CSV 或场景目录中可发现
- **THEN** 系统 MUST 将对应 path 写入 manifest
- **AND** path 不可用时对应 manifest 列 MUST 为空或标记不可用

#### Scenario: 缺失模态不阻断 GPS baseline
- **WHEN** 某 optional modality 在全部或部分场景缺失
- **THEN** manifest builder MUST 继续完成
- **AND** residual training MUST 仍能运行 `gps_prior_only` 与 `gps_context_only_residual`
- **AND** skipped modality ablation MUST 在 summary 中记录 `skipped_reason`

### Requirement: Residual dataset 按启用模态读取
Residual fusion Dataset/DataLoader MUST 根据 manifest 与 ablation 启用模态读取数据，不得读取未启用或不可用的 optional modality。

#### Scenario: GPS context only 不读取 sensor 文件
- **WHEN** ablation 为 `gps_context_only_residual`
- **THEN** Dataset MUST 只读取 GPS context、prior logits/stats 和标签所需字段
- **AND** Dataset MUST 不读取 image、LiDAR 或 radar 文件

#### Scenario: array modality shape 校验
- **WHEN** LiDAR 或 radar array feature 被启用
- **THEN** Dataset MUST 校验 array shape 是否可被选定 encoder 处理
- **AND** shape 不一致且无法使用预处理 feature 时 MUST 报告清晰错误或跳过该 ablation

### Requirement: Camera residual manifest data loading
系统 MUST 支持从 camera residual manifest 构建 Dataset/DataLoader。该数据加载路径 MUST 按当前 stage 和 ablation 只读取需要的 image 或 AE feature，并在 image 缺失时保持 GPS-only baseline 可运行。

#### Scenario: gps_prior_only 不读取 image
- **WHEN** ablation 为 `gps_prior_only`
- **THEN** Dataset MUST NOT 读取 image 文件
- **AND** Dataset MUST NOT 要求 `ae_feature_path` 存在
- **AND** 样本 MUST 仍包含 GPS prior、GPS pred、GPS context、target label 和 split role

#### Scenario: AE training 跳过 missing image
- **WHEN** AE training Dataset 读取 manifest
- **THEN** Dataset MUST 只使用 `image_exists=true` 的样本
- **AND** 没有任何可用 image 时 MUST 抛出清晰错误

#### Scenario: residual training 使用 AE feature
- **WHEN** ablation 需要 `camera_ae_feature`
- **THEN** Dataset MUST 根据 `ae_feature_path` 和 `ae_feature_row_index` 读取 feature
- **AND** feature 不可用的样本 MUST 按配置跳过或降级为 GPS context only
- **AND** 降级或跳过原因 MUST 写入 run metadata 或 summary

#### Scenario: query label 不进入训练 batch
- **WHEN** Dataset 为 train/support loader 构建 batch
- **THEN** target query 样本 MUST 不进入训练 batch
- **AND** 如果 evaluation loader 包含 query label，loss 计算 MUST 使用 evaluation-only 路径，不得反向传播或用于 early stopping

### Requirement: 模态数据加载不依赖蒸馏配置
Dataset、batch preparation 和 label 对齐 MUST 由 experiment task、enabled modalities、prediction objective 和 supervised/adaptation workflow 决定。数据加载层 MUST 不读取 `distillation` 配置来决定 batch 字段或 label 语义。

#### Scenario: batch 构建忽略 distillation 字段
- **WHEN** 用户运行任一 supported supervised/adaptation 配置
- **THEN** batch preparation MUST 只根据 task 和 enabled modalities 构造输入
- **AND** 配置中若出现 `distillation` 字段 MUST 在配置解析阶段失败

### Requirement: DeepSense6G Top8 selector optional modality loading
DeepSense6G Top8 candidate dataset MUST 按配置和 manifest availability 加载 optional modalities。未启用或不可用的 camera AE、image tensor、LiDAR feature 和 radar feature MUST 不阻止 GPS context-only selector 运行；启用某个 optional modality 时，dataset MUST 只读取该模态需要的 path 或 feature，不触发其它模态 IO。

#### Scenario: GPS context-only selector 不读取图像或点云
- **WHEN** 配置运行 `gps_context_only_selector`
- **THEN** dataset MUST 读取 Top8 candidate manifest、candidate fields 和 GPS context fields
- **AND** dataset MUST NOT 读取 image file、camera AE feature、LiDAR feature 或 radar feature
- **AND** 返回样本 MUST 不包含未启用 optional modality 的大张量

#### Scenario: camera AE 可用时按 row index 读取
- **WHEN** 配置启用 camera AE feature 且 manifest 包含有效 `camera_ae_feature_row_index`
- **THEN** dataset MUST 从配置的 AE feature artifact 读取对应 feature row
- **AND** 返回样本 MUST 包含 `camera_ae_feature`
- **AND** dataset MUST 在 metadata 中记录 AE feature artifact path 或 fingerprint

#### Scenario: camera AE 缺失时记录原因
- **WHEN** 配置启用 camera AE feature 但 manifest 中 feature row index 无效或 artifact 缺失
- **THEN** dataset MUST 返回缺失标记
- **AND** runner MUST 跳过 camera AE 相关 ablation 或降级到 GPS context-only selector
- **AND** summary MUST 写入 `skipped_reason`

#### Scenario: image/LiDAR/radar feature 按需读取
- **WHEN** 配置启用 image tensor、LiDAR feature 或 radar feature
- **THEN** dataset MUST 只读取对应模态字段中声明的 path 或 feature
- **AND** 其它未启用模态 MUST 不触发 path 解析、cache 初始化或文件读取

### Requirement: Top8 selector normalization fit boundary
Top8 selector dataset MUST 支持为 candidate features 和 GPS context 保存 normalization metadata。E、N、log_range、speed、candidate logits 等统计量 MUST 只从允许训练的 source/support 样本拟合，target query 样本 MUST 不参与 fit。

#### Scenario: support/source fit scaler
- **WHEN** dataset 构建 normalization artifact
- **THEN** scaler fit MUST 只使用 source training rows、target support rows 或 target support internal train rows
- **AND** metadata MUST 记录 fit split、样本数、字段名和随机种子

#### Scenario: query 不参与 normalization fit
- **WHEN** manifest 中包含 target query rows
- **THEN** target query rows MUST 只使用已经拟合好的 normalization 参数进行 transform
- **AND** target query label 或 query 统计量 MUST NOT 影响 scaler 参数

### Requirement: GPS+LiDAR BGAM 按需模态加载
GPS+LiDAR BGAM dataset MUST 按配置和 manifest availability 加载 GPS prior、TopK candidates 和 LiDAR 输入。未启用 LiDAR、image、camera AE 或 radar 时，dataset MUST 不触发对应模态 IO；GPS-only ablation MUST 不读取 LiDAR 点云或 BEV cache。

#### Scenario: gps_only 不读取 LiDAR
- **WHEN** 配置运行 `gps_only` ablation
- **THEN** dataset MUST 只读取 BGAM manifest 中的 GPS prior、candidate beams/probs 和 label/evaluation metadata
- **AND** dataset MUST NOT 读取 raw LiDAR point cloud、LiDAR BEV cache、image、camera AE 或 radar feature

#### Scenario: BGAM ablation 按需读取 LiDAR
- **WHEN** 配置运行包含 BGAM 或 LiDAR 的 ablation
- **THEN** dataset MUST 读取当前样本所需的 `lidar_bev_cache_path` 或 `lidar_path`
- **AND** dataset MUST NOT 读取未启用的 image、camera AE 或 radar feature
- **AND** LiDAR 读取 MUST 发生在取样阶段而不是 dataset 初始化阶段

#### Scenario: LiDAR 缺失时记录 skipped reason
- **WHEN** 配置启用 LiDAR ablation 但 manifest 行缺少 LiDAR path 或文件不存在
- **THEN** 系统 MUST 早失败或按配置跳过该 ablation
- **AND** summary/run metadata MUST 写入 `skipped_reason`、缺失字段和受影响样本数

### Requirement: GPS+LiDAR BGAM 防泄漏数据边界
GPS+LiDAR BGAM 数据构建 MUST 区分训练输入、loss label 和最终评价字段。future ground-truth beam label MUST 只作为 loss/evaluation target；target query rows MUST 不参与 normalization fit、mask construction、early stopping 或 checkpoint selection。

#### Scenario: target label 不进入模型输入
- **WHEN** dataset 返回一个训练或评估样本
- **THEN** `gt_beam` 或 `target_label` MUST 单独作为 label 字段返回
- **AND** 模型输入字段 MUST 不包含由 target label 派生的 BGAM mask、AoD prior、candidate probability 或 LiDAR feature

#### Scenario: query 不参与 normalizer fit
- **WHEN** BGAM dataset 或 runner fit GPS/LiDAR/candidate normalizer
- **THEN** fit rows MUST 只来自 source train、target support 或 target support internal train split
- **AND** target query rows MUST 只使用已 fit 的 normalizer transform
- **AND** metadata MUST 记录 `query_label_used_for_training=false`

### Requirement: GPS+LiDAR BGAM manifest column mapping
BGAM manifest loader MUST 支持配置化字段名映射，以兼容 Top8 manifest、DeepSense6G sequence CSV 和用户提供的 GPS+LiDAR manifest。字段映射 MUST 输出统一内部字段，并 MUST 在缺失必要字段时给出清晰错误。

#### Scenario: local coordinate columns
- **WHEN** manifest 提供 local coordinate columns
- **THEN** loader MUST 按配置映射为 `user_x`、`user_y`、`rsu_x`、`rsu_y` 和 `rsu_yaw`
- **AND** loader MUST 使用这些字段生成 `theta_gps` 和 `distance_to_rsu`

#### Scenario: GPS logits/probs columns
- **WHEN** manifest 提供 `gps_prob_0` 到 `gps_prob_63` 或 `gps_logits_path`
- **THEN** loader MUST 读取或构造 `[64]` GPS prior tensor
- **AND** loader MUST 从该 prior 生成 TopK candidates 或校验与 manifest candidates 一致

