# training-throughput-optimization Specification

## Purpose
定义训练吞吐 profiling、DataLoader/cache/transfer 参数建议和并行运行边界。
## Requirements
### Requirement: 训练吞吐 profiling
项目 MUST 提供可独立运行的训练 I/O 和 step profiling 入口，用于定位 Scenario 9 dataset、DataLoader、CPU 到 GPU transfer 和模型训练 step 的耗时。profile 入口 MUST 可通过配置文件和命令行覆盖控制样本数、split、device、warmup 和输出路径。

#### Scenario: 输出 dataset 与 DataLoader 耗时
- **WHEN** 用户运行 profile 入口并指定一个训练配置
- **THEN** 系统 MUST 记录 dataset `__getitem__` 的均值、P50 和 P95 耗时
- **AND** 系统 MUST 记录 DataLoader batch wait 的均值、P50 和 P95 耗时
- **AND** 输出 MUST 包含实际启用模态、样本数、batch size、num_workers、pin_memory、persistent_workers 和 prefetch_factor

#### Scenario: 输出 GPU step 分解
- **WHEN** 用户在 CUDA device 上运行 profile 入口
- **THEN** 系统 MUST 记录 CPU 到 GPU transfer、forward、backward、optimizer step 和总 step 耗时
- **AND** 系统 MUST 记录 samples/s 和 CUDA peak memory
- **AND** GPU 计时 MUST 使用同步或 CUDA event，避免异步执行导致明显低估

#### Scenario: CPU-only profile 可运行
- **WHEN** 用户在 CPU 或 `device=cpu` 下运行 profile 入口
- **THEN** 系统 MUST 跳过 CUDA memory 和 CUDA event 指标
- **AND** profile MUST 仍输出 dataset、DataLoader、forward/loss 和总 step 耗时

### Requirement: Beam label cache
Scenario 9 dataset MUST 避免对重复 beam label 文本执行重复 `np.loadtxt + argmax`。系统 MUST 支持为当前 split 的唯一 beam path 建立轻量 path-to-label cache，或按需 lazy cache。beam label cache MUST 不改变 `input_beam` 和 `target_beam` 的张量维度、dtype 和语义。

#### Scenario: 重复 beam path 只解析一次
- **WHEN** 多个滑动窗口引用同一个 beam 文本路径
- **THEN** dataset SHOULD 只解析该路径一次并复用整数 label
- **AND** 后续样本 MUST 从 cache 获取相同 label

#### Scenario: beam 文件错误清晰报告
- **WHEN** 启用 beam label cache 时遇到缺失、空文件或无法解析的 beam 文本
- **THEN** 系统 MUST 抛出包含 beam 路径的清晰错误
- **AND** 系统 MUST 不返回静默错误 label

### Requirement: Non-blocking transfer 与 AMP
训练和验证 batch 准备 MUST 支持可配置的 non-blocking tensor transfer。训练流程 MUST 支持可配置 AMP，并在 CUDA 可用且 AMP 启用时使用 autocast 和 GradScaler；当 AMP 禁用或 device 不是 CUDA 时，系统 MUST 保持 FP32 训练路径兼容。

#### Scenario: pinned memory 使用 non-blocking transfer
- **WHEN** DataLoader 配置 `pin_memory: true` 且训练配置启用 non-blocking transfer
- **THEN** batch 准备函数 MUST 使用 `tensor.to(device, non_blocking=True)` 或等价调用
- **AND** labels、image、radar、GPS 和 LiDAR 输入 MUST 使用一致的 transfer 配置

#### Scenario: AMP CUDA 训练
- **WHEN** 用户在 CUDA device 上设置 `training.amp.enabled: true`
- **THEN** 训练 forward 和 loss 计算 MUST 在 autocast 上下文中执行
- **AND** backward 和 optimizer step MUST 使用 GradScaler 或配置指定的安全缩放策略
- **AND** checkpoint、metrics 和日志结构 MUST 保持兼容

#### Scenario: AMP 禁用回退
- **WHEN** 用户设置 `training.amp.enabled: false` 或 device 不是 CUDA
- **THEN** 训练 MUST 使用现有 FP32 路径
- **AND** 不得要求 CUDA AMP 组件才能完成 CPU smoke test

### Requirement: Image 路径不产生 motion cache
训练吞吐优化 MUST 不再依赖 image motion mask cache。包含 image modality 的 profile、训练和评估运行 MUST 直接衡量 RGB/ImageNet image 加载与模型 step，而不是预热、读取或写入 image motion cache。

#### Scenario: image profile 不报告 motion cache
- **WHEN** 用户运行包含 image modality 的训练 I/O profile
- **THEN** profile 输出 MUST 记录 image modality 使用 RGB/ImageNet 输入
- **AND** profile 输出 MUST 不包含 image motion cache hit、miss、write 或 cache 目录指标

#### Scenario: 旧 image motion 预处理入口不可用
- **WHEN** 用户运行旧 image motion mask cache 预处理入口或加载已退役的 `configs/preprocess/image_motion_cache.yaml`
- **THEN** 系统 MUST 拒绝该操作
- **AND** 错误信息 MUST 说明 image motion cache 已删除且不会生成替代 cache

### Requirement: 并行训练吞吐诊断
训练吞吐 profile MUST 支持定位多实验并行时的 GPU 低利用率原因。profile 输出 MUST 包含 DataLoader wait 分布、模态级样本加载耗时、train/test DataLoader worker 数量、persistent worker 状态、prefetch 配置、progress 输出状态和启用 cache 策略。

#### Scenario: 输出并行训练瓶颈字段
- **WHEN** 用户对五模态 fusion 配置运行训练吞吐 profile
- **THEN** profile 输出 MUST 记录每个启用模态的 `__getitem__` 耗时汇总
- **AND** 输出 MUST 记录 train/test split 的 batch size、num_workers、persistent_workers、prefetch_factor 和 pin_memory
- **AND** 输出 MUST 记录 `output.progress.enabled` 和 cache policy

#### Scenario: 识别 DataLoader 等待尖峰
- **WHEN** DataLoader batch wait 的 P95 明显高于 GPU forward/backward 总耗时
- **THEN** profile 输出 MUST 能通过结构化字段表现该差异
- **AND** profile 文档 MUST 指引用户优先检查 worker 数、cache 和后台日志，而不是只检查模型结构

### Requirement: 并行训练配置推荐
项目 MUST 提供并行训练配置建议，使用户能够根据并行实验数、CPU 数、启用模态和 cache 状态生成安全的命令行覆盖参数。推荐结果 MUST 不直接修改用户配置文件，除非用户明确执行实现任务中的配置落盘步骤。

#### Scenario: 四实验五模态推荐
- **WHEN** 用户请求为 4 个五模态训练生成推荐配置
- **THEN** 系统 MUST 推荐限制每个训练的 DataLoader worker、提高或保持合理 prefetch、关闭后台 batch 级 progress、复用 LiDAR cache，并可选启用 AMP
- **AND** 推荐 MUST 说明这些覆盖参数适用于后台并行训练而非所有单实验默认值

#### Scenario: cache 未预热提示
- **WHEN** 推荐器发现 LiDAR cache 不存在或覆盖率不足
- **THEN** 系统 MUST 输出 cache 预热建议
- **AND** 不得默认建议 `data.cache.policy=read_only` 作为唯一方案

### Requirement: DataLoader worker 生命周期控制
训练流程 MUST 支持为 train/test split 分别设置 DataLoader worker 和 persistent worker 策略，避免验证用 worker 在整个训练期间无条件占用 CPU。现有统一 `data.dataloader.num_workers` 配置 MUST 保持兼容。

#### Scenario: test loader 使用独立 worker 配置
- **WHEN** 配置提供 test split 专用 worker 参数
- **THEN** test DataLoader MUST 使用该参数构建
- **AND** train DataLoader MUST 继续使用 train split 参数

#### Scenario: 旧配置兼容
- **WHEN** 配置只提供现有 `data.dataloader.num_workers`
- **THEN** train/test DataLoader MUST 继续按旧行为构建
- **AND** 现有训练、验证和 smoke test MUST 不需要修改配置即可运行

### Requirement: 后台训练日志降噪
训练流程 MUST 支持后台低噪声 progress 模式，减少 batch 级 tqdm 输出写入 tmux/tee 日志。禁用或降频 progress MUST 不影响 TensorBoard、epoch history、checkpoint 或 `training_outputs.npz`。

#### Scenario: 禁用 batch progress
- **WHEN** 用户设置 `output.progress.enabled=false`
- **THEN** 训练 MUST 不输出 batch 级 tqdm 进度条
- **AND** epoch metrics、TensorBoard scalar、checkpoint 和最终训练输出 MUST 正常生成

#### Scenario: 并行推荐默认降噪
- **WHEN** 并行训练推荐器生成后台 tmux 命令覆盖参数
- **THEN** 推荐参数 MUST 包含关闭或降低 batch progress 输出的设置
- **AND** 推荐说明 MUST 提供查看日志和 TensorBoard 的替代方式

### Requirement: 训练 epoch 级 train 子采样
训练流程 MUST 支持显式配置的 train epoch 子采样，使用户能够在保留原 train CSV 和 dataset 语义的前提下限制每个 epoch 实际参与训练的样本数。该能力 MUST 默认关闭；关闭时训练 MUST 继续遍历完整 train split。

#### Scenario: 默认完整 train split
- **WHEN** 用户未启用 `training.epoch_subsampling.enabled`
- **THEN** train DataLoader MUST 按现有行为遍历完整 train dataset
- **AND** 现有训练配置 MUST 不需要修改即可运行

#### Scenario: 按比例限制每个 epoch 样本
- **WHEN** 用户设置 `training.epoch_subsampling.enabled=true` 且提供合法 `fraction`
- **THEN** 每个 train epoch MUST 使用完整 train dataset 中按该比例计算的有效样本数
- **AND** 有效样本数 MUST 至少为 1 且不得超过完整 train dataset 长度

#### Scenario: 按固定数量限制每个 epoch 样本
- **WHEN** 用户设置 `training.epoch_subsampling.enabled=true` 且提供合法 `num_samples`
- **THEN** 每个 train epoch MUST 使用不超过 `num_samples` 的 train 样本
- **AND** 当 `num_samples` 大于完整 train dataset 长度时，系统 MUST 退化为完整 train epoch 并在运行元数据中记录该退化结果

#### Scenario: 子采样配置错误清晰失败
- **WHEN** 用户同时设置 `fraction` 和 `num_samples` 或提供非法比例/数量
- **THEN** 训练启动 MUST 失败并给出包含 `training.epoch_subsampling` 的清晰错误信息
- **AND** 系统 MUST 不静默回退为完整训练

#### Scenario: 每 epoch 可复现轮换抽样
- **WHEN** train epoch 子采样启用且 `rotate_each_epoch=true`
- **THEN** 不同 epoch MUST 基于实验 seed 和 epoch 编号生成可复现的无放回样本选择
- **AND** checkpoint resume 后同一绝对 epoch MUST 生成与未中断运行一致的样本选择

#### Scenario: 固定子集调试
- **WHEN** train epoch 子采样启用且 `rotate_each_epoch=false`
- **THEN** 每个 epoch MUST 使用同一个可复现 train 子集
- **AND** 该子集 MUST 由配置 seed 或 `experiment.seed` 决定

#### Scenario: 验证 split 不受 train 子采样影响
- **WHEN** train epoch 子采样启用并完成一个训练 epoch
- **THEN** 验证或测试 DataLoader MUST 继续使用完整 validation/test split
- **AND** 验证指标 MUST 不因 train 子采样配置而减少评估样本数

#### Scenario: 运行产物记录子采样语义
- **WHEN** train epoch 子采样启用
- **THEN** 最终配置、运行元数据或训练日志 MUST 记录完整 train split 样本数、每个 epoch 有效 train 样本数、抽样方式、seed、`rotate_each_epoch` 和是否退化为完整 epoch
- **AND** epoch 级日志 MUST 能区分完整训练 epoch 与子采样训练 epoch

### Requirement: MMW image-heavy 吞吐 profile
训练吞吐 profile MUST 支持当前 MMW image-heavy workflow 或显式 fixture 配置，并输出足以定位 RGB/ImageNet image 序列解码、DataLoader wait、worker 内存和 GPU step 关系的结构化字段。profile MUST 不把 image decode/resize 等待误判为 CUDA 显存或模型结构问题，且 MUST 不要求退役的 MMW HiST-Beam LOSO 配置存在。

#### Scenario: 输出 MMW image-heavy profile 字段
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-training-throughput --mode profile --config <current-mmw-image-heavy-config-or-fixture>`
- **THEN** profile 输出 MUST 记录 enabled modalities、seq_len、batch size、num_workers、prefetch_factor、pin_memory 和 persistent_workers
- **AND** profile 输出 MUST 记录 image、GPS、mmWave、beam label 和允许的 beam power 或 dataset metadata 相关 `__getitem__` 耗时汇总
- **AND** profile 输出 MUST 记录 DataLoader wait、transfer、forward、backward/optimizer、step、samples/s 和 CUDA peak memory

#### Scenario: 标记 loader wait 支配 step
- **WHEN** DataLoader wait 的 P95 或总耗时显著高于 GPU step
- **THEN** profile 输出 MUST 标记 `loader_wait_dominates_step`
- **AND** 输出 MUST 指出优先检查 image decode/cache、batch size、worker 数和并行 run 数

### Requirement: MMW 并行训练内存风险推荐
并行训练推荐器 MUST 对当前 MMW image-heavy 运行输出 memory-aware 覆盖建议。推荐 MUST 同时考虑 enabled modalities、seq_len、batch size、DataLoader worker 数、parallel run 数、系统内存和已有 profile 结果。

#### Scenario: image-heavy 并行运行推荐保守 worker
- **WHEN** 用户为包含 image modality 且 `seq_len >= 8` 的当前 MMW 配置请求 4 路并行推荐
- **THEN** 推荐 MUST 限制每个 run 的 train worker 和 batch size，使预计 worker RSS 不超过可用内存预算
- **AND** 推荐 MUST 说明 AMP 不能解决 image decode 或 worker RSS 问题
- **AND** 推荐 MUST 包含 `output.progress.enabled=false` 或等价后台低噪声设置

#### Scenario: OOM 风险给出降载路径
- **WHEN** profile 或运行日志显示进程被系统 killed、退出码 137 或 worker RSS 过高
- **THEN** 推荐 MUST 输出降低 parallel runs、降低 batch size、降低 num_workers、禁用 persistent workers 或启用 image-derived cache 的建议
- **AND** 推荐 MUST 不默认建议继续增加 worker 数

### Requirement: MMW throughput 回归验证
项目 MUST 提供 focused 验证，确保 MMW image-heavy profile 和推荐字段稳定，并能在无全量真实数据提交的情况下运行。

#### Scenario: profile 字段稳定
- **WHEN** 测试或小样本 fixture 运行 MMW image-heavy profile helper
- **THEN** 输出 MUST 包含分模态 getitem、DataLoader wait、GPU/CPU step、loader config、seq_len、enabled modalities 和 IO-risk 字段
- **AND** 测试 MUST 可通过 `conda run -n kd_mm_beam pytest ... -q` 运行

#### Scenario: 推荐器识别 image-heavy 风险
- **WHEN** 推荐器收到包含 image modality、`seq_len=8` 和多个并行 run 的 MMW 配置
- **THEN** 推荐结果 MUST 包含 memory risk 或 image-heavy risk 诊断
- **AND** 推荐覆盖参数 MUST 优先限制 worker、batch size 或并行度，而不是只启用 AMP

### Requirement: 吞吐优化配置与日志
训练配置 MUST 暴露吞吐相关开关，包括 DataLoader worker/prefetch 参数、non-blocking transfer、AMP 和预处理 cache 读取/写入策略。训练日志、最终配置或 profile 输出 MUST 记录这些实际生效的吞吐参数，便于比较不同实验设置。

#### Scenario: 记录吞吐参数
- **WHEN** 一次训练或 profile 运行启动
- **THEN** 输出配置或日志 MUST 记录 `num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor`、non-blocking transfer、AMP enabled/dtype 和启用的 cache 目录
- **AND** 对启用 image 或 LiDAR 的配置 MUST 记录对应 cache 参数 hash 目录

#### Scenario: 并行实验默认不过度放大 worker
- **WHEN** 用户使用 canonical 单模态或 fusion YAML 运行实验
- **THEN** 配置 SHOULD 使用适合并行实验的保守 `num_workers` 和 `prefetch_factor`
- **AND** 用户 MUST 能通过命令行覆盖这些参数以寻找单实验最高吞吐

### Requirement: Training throughput profiling 与推荐必须共享 owner
Training IO profiling、瓶颈汇总和 parallel training recommendation MUST 由一个 throughput owner 或 package CLI mode 管理。项目 SHOULD 不保留独立 recommendation script 重复解析配置、硬件或 profiling fields。

#### Scenario: recommendation 读取 profiling owner 输出
- **WHEN** 协作者需要 parallel training recommendation
- **THEN** 推荐逻辑 SHOULD 读取 profiling owner 的 output summary 或由同一 owner mode 计算
- **AND** MUST 不复制 profiling input discovery、config parsing 或 metrics formatting 逻辑

#### Scenario: profile script 合并后字段保持
- **WHEN** `scripts/profile_training_io.py` 行为迁移到 package CLI 或 owner module
- **THEN** sampling fields、throughput metrics、IO bottleneck labels 和 recommendation inputs MUST 保持稳定或同步更新 current spec
- **AND** focused tests MUST 覆盖 profiling output 和 recommendation mode

### Requirement: Throughput wrapper 保留必须说明独立价值
若 throughput profiling 或 recommendation wrapper 因外部复现实验仍需保留，项目 MUST 在 inventory 或 current spec 中记录 retained-with-reason。

#### Scenario: 保留脚本有删除触发条件
- **WHEN** throughput wrapper 暂时保留
- **THEN** retained-with-reason MUST 包含独立契约、替代 owner 缺口和未来删除触发条件
- **AND** docs MUST 不把 wrapper 描述为优先于 owner CLI 的推荐入口

