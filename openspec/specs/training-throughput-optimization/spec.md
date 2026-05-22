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
- **WHEN** 用户运行 image motion mask cache 预处理入口或加载 `configs/preprocess/image_motion_cache.yaml`
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
