## ADDED Requirements

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

### Requirement: Image motion mask cache
系统 MUST 支持将 image motion mask 预处理结果缓存到磁盘，并在训练和评估 dataset 中按需复用。cache MUST 通过预处理参数 hash 或等价机制隔离 image size、Gaussian sigma、阈值策略、灰度化方式和 cache version。cache 读取 MUST 保持懒加载，不得在 Dataset 初始化阶段读取全部 mask 数组。

#### Scenario: 从 cache 读取 motion mask
- **WHEN** image modality 启用且配置启用 `image_motion_use_cache`
- **AND** 当前相邻帧 pair 的 cache 文件存在且参数匹配
- **THEN** dataset MUST 从 cache 读取该 pair 的 motion mask
- **AND** 返回的 image tensor shape 和 dtype MUST 与在线计算路径兼容

#### Scenario: cache miss 在线生成并写入
- **WHEN** image modality 启用且配置启用 `image_motion_write_cache`
- **AND** 当前相邻帧 pair 的 cache 文件不存在
- **THEN** dataset MAY 在线读取原始 jpg 并生成 motion mask
- **AND** 系统 MUST 将生成结果写入当前参数 hash cache 目录
- **AND** 当前样本的 label、序列长度和返回字段 MUST 不改变

#### Scenario: 参数变化不误用旧 cache
- **WHEN** `image_size`、Gaussian sigma、阈值策略、灰度化方式或 cache version 改变
- **THEN** 系统 MUST 使用不同 cache 目录、cache key 或 metadata 校验结果
- **AND** 系统 MUST 不默认复用旧参数生成的 motion mask

#### Scenario: 预处理入口预热 image cache
- **WHEN** 用户运行 image motion mask cache 预处理入口并提供 train/test CSV
- **THEN** 系统 MUST 收集唯一相邻帧 pair 并为缺失 pair 生成 cache
- **AND** 系统 MUST 跳过已存在且参数匹配的 cache 文件
- **AND** 系统 MUST 写出包含参数、生成数量、跳过数量和源 CSV 的 metadata

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
