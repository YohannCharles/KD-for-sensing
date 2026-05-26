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

### Requirement: Multimodal-NF 吞吐 profile
训练吞吐 profile MUST 支持 Multimodal-NF image、LiDAR、GPS、CSI 和 fusion 配置，并输出足以定位 image/LiDAR HDF5 解压、派生缓存、DataLoader wait、CPU 到 GPU transfer 和模型 step 的结构化字段。

#### Scenario: 输出 Multimodal-NF 模态级 getitem
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/profile_training_io.py --config <multimodal-nf-config>`
- **THEN** profile 输出 MUST 记录每个启用 Multimodal-NF 模态的 `__getitem__` 均值、P50、P95、最小值和最大值
- **AND** 输出 MUST 包含 dataset 总 `__getitem__`、auxiliary targets、DataLoader wait、transfer、forward、backward/optimizer、step、samples/s 和 CUDA peak memory 字段

#### Scenario: 输出 Multimodal-NF 数据源和缓存策略
- **WHEN** profile 的配置使用 Multimodal-NF dataset
- **THEN** profile 输出 MUST 记录每个 split 和启用模态的数据源类型、派生缓存策略、缓存路径、缓存命中或缺失状态
- **AND** 如果 profile 过程中触发缓存生成，输出 MUST 记录生成行为或生成耗时摘要

#### Scenario: 对比 GPS 与 image/LiDAR 瓶颈
- **WHEN** 用户分别 profile Multimodal-NF GPS-only、image-only、LiDAR-only 和 fusion 配置
- **THEN** 输出字段 MUST 足以比较各模态 `__getitem__` 与 GPU step 的相对耗时
- **AND** profile MUST 不把 image/LiDAR 的数据读取耗时归入 GPS 或 target 字段

### Requirement: Multimodal-NF 吞吐配置推荐
项目 MUST 提供面向 Multimodal-NF image/LiDAR/fusion 训练的配置推荐，帮助用户选择 DataLoader worker、prefetch、pin memory、test worker、AMP、progress 和派生缓存策略。推荐 MUST 以命令行覆盖或说明形式输出，除非用户明确要求，不得直接修改用户配置文件。

#### Scenario: 含 image 和 LiDAR 的 fusion 推荐
- **WHEN** 用户请求 Multimodal-NF image+LiDAR+GPS fusion 的吞吐推荐
- **THEN** 推荐 MUST 包含启用或预热 image/LiDAR 派生缓存的建议
- **AND** 推荐 MUST 包含合理的 `num_workers`、`prefetch_factor`、`pin_memory`、`persistent_workers`、`test_num_workers`、`training.amp.enabled` 和 `output.progress.enabled` 覆盖建议
- **AND** 推荐 MUST 说明这些建议用于吞吐优化，用户仍可按机器资源调整

#### Scenario: 缓存未准备时提示预热
- **WHEN** 推荐器或 profile metadata 发现 Multimodal-NF image/LiDAR 派生缓存不存在或覆盖率不足
- **THEN** 输出 MUST 提示先运行缓存预热或使用 `auto`/`rebuild` 策略
- **AND** 输出 MUST 不默认建议把缺失缓存的配置强行改成 `read_only`

#### Scenario: GPS-only 不推荐重缓存
- **WHEN** 用户运行或请求 GPS-only Multimodal-NF 配置推荐
- **THEN** 推荐 MUST 不要求 image 或 LiDAR 派生缓存
- **AND** 推荐 MUST 说明 GPS-only 主要受模型 step 和普通 DataLoader 参数影响

### Requirement: Multimodal-NF 吞吐回归验证
实现 Multimodal-NF 吞吐优化后，项目 MUST 提供 focused 验证，确保派生缓存路径不改变样本契约，并且 profile 能捕获 image/LiDAR 的性能差异。验证 MUST 使用 fixture 或小样本本地数据，不能要求提交真实全量缓存。

#### Scenario: 派生缓存与原始读取等价
- **WHEN** 测试使用小型 Multimodal-NF fixture 同时构建原始 HDF5 dataset 和派生缓存 dataset
- **THEN** 两条路径返回的 image/LiDAR tensor shape、target fields、metadata 关键字段和样本顺序 MUST 等价
- **AND** 未启用模态 MUST 不出现在样本中

#### Scenario: profile 输出字段稳定
- **WHEN** 测试运行 focused profile 或直接调用 profile helper
- **THEN** 输出 MUST 包含 Multimodal-NF 模态级 getitem、数据源、缓存策略、DataLoader split 参数和 samples/s 字段
- **AND** 该测试 MUST 可通过 `conda run -n kd_mm_beam pytest ... -q` 在无真实全量数据时运行

#### Scenario: 配置兼容性
- **WHEN** 用户未配置 Multimodal-NF 派生缓存字段
- **THEN** 现有 Multimodal-NF 配置 MUST 继续构建 dataset 和 DataLoader
- **AND** focused tests MUST 覆盖旧配置默认行为

### Requirement: Multimodal-NF cache IO profiling
训练吞吐 profile MUST 对 Multimodal-NF image/LiDAR 派生 cache 的初始化、校验、打开和读取耗时提供结构化诊断。profile 输出 MUST 能区分 cache validation、DataLoader wait、CPU 到 GPU transfer、forward/backward 和 optimizer step，避免把 IO 等待误判为 CUDA 或模型结构问题。

#### Scenario: 输出 cache 校验耗时
- **WHEN** 用户对 Multimodal-NF image/LiDAR/fusion 配置运行训练吞吐 profile
- **THEN** profile 输出 MUST 记录每个启用派生 cache 模态的 validation mode、validation duration、是否执行 source fingerprint scan、cache policy 和 source kind
- **AND** profile 输出 MUST 记录 dataset 构建或 cache plan 解析耗时

#### Scenario: 输出 cache open/read 耗时
- **WHEN** profile 读取 Multimodal-NF image/LiDAR 样本
- **THEN** profile 输出 MUST 记录 cache open 次数、cache read 次数、cache read 平均值、P95、最大值和按模态汇总的 `__getitem__` 耗时
- **AND** 输出 MUST 包含 cache path 数量、cache 总字节数、storage kind、layout 和推荐访问模式

#### Scenario: 输出 IO-risk 判定
- **WHEN** DataLoader wait 的 P95 明显高于模型 step 或 cache read 尾延迟明显高于均值
- **THEN** profile 输出 MUST 用结构化字段标记 IO-risk，例如随机读风险、loader wait dominates step、cache validation scan detected 或 mmap page fault risk
- **AND** profile 输出 MUST 保留原有 samples/s、CUDA peak memory、worker、prefetch、pin memory、AMP 和 progress 字段

### Requirement: Multimodal-NF train 子采样局部性控制
训练流程 MUST 支持或推荐 Multimodal-NF image/LiDAR 大 cache 场景下的局部性优先 train 子采样顺序。该能力 MUST 保持样本选择可复现，并 MUST 在 run metadata 中记录完整 train 样本数、有效样本数、seed、轮换策略和输出顺序策略。

#### Scenario: 随机选择后按局部性排序
- **WHEN** train epoch 子采样启用且用户选择局部性优先顺序
- **THEN** 每个 epoch MUST 先基于 seed 和 epoch 生成可复现的无放回样本子集
- **AND** 输出给 DataLoader 的样本顺序 MUST 按 dataset index、source key、block key 或等价局部性键排序或分块
- **AND** 有效样本集合 MUST 与同 seed 下的随机选择语义一致，除非配置显式选择固定顺序调试模式

#### Scenario: 旧随机顺序兼容
- **WHEN** 用户未配置局部性优先顺序且保持现有 `shuffle=true`
- **THEN** train 子采样 MUST 保持现有随机顺序语义
- **AND** 现有训练配置 MUST 不需要修改即可运行

#### Scenario: 运行产物记录局部性策略
- **WHEN** 局部性优先顺序被启用或由推荐器输出
- **THEN** 最终配置、运行元数据或 epoch 日志 MUST 记录排序策略、是否 block shuffle、block size 或等价参数
- **AND** 用户 MUST 能区分完整随机 batch 顺序和 IO-friendly batch 顺序

### Requirement: Multimodal-NF 并行训练 IO 推荐
并行训练推荐器 MUST 对 Multimodal-NF image/LiDAR/fusion 运行输出 IO-aware 覆盖参数和说明。推荐结果 MUST 不直接修改用户配置文件，MUST 明确区分后台并行训练建议与单实验默认值。

#### Scenario: image/LiDAR/fusion 后台并行推荐
- **WHEN** 用户为 Multimodal-NF image/LiDAR/fusion 配置请求并行训练推荐
- **THEN** 推荐器 MUST 输出关闭 batch progress、合理 worker/prefetch、cache policy、cache validation mode 和局部性优先 train 子采样顺序建议
- **AND** CUDA image/fusion 训练推荐 MUST 包含 AMP 覆盖或明确说明 AMP 适用条件
- **AND** 推荐说明 MUST 提醒用户避免把 `read_only` cache 等同于必然高速路径

#### Scenario: cache warm 后避免重复强校验
- **WHEN** 推荐器判断 Multimodal-NF image/LiDAR 派生 cache 已预热且 sidecar 可轻量校验
- **THEN** 推荐器 MUST 建议 warm cache 训练使用轻量运行时校验
- **AND** 推荐器 MUST 不建议每个并行训练进程重复执行原始 HDF5 全量 fingerprint scan

#### Scenario: GPU 分配和重 IO run 提示
- **WHEN** 用户请求多个 Multimodal-NF image/LiDAR/fusion 后台训练并行运行
- **THEN** 推荐输出 MUST 包含重 IO run 的 GPU 分配提示，建议优先跨 GPU 均匀分配并避免在同一 GPU 上叠加多个重 IO run
- **AND** 当并行数超过 GPU 数或 cache IO-risk 较高时，推荐输出 MUST 提示降低并行度、启用局部性顺序或先运行小样本 profile

#### Scenario: profile 驱动推荐
- **WHEN** 用户提供或生成了 Multimodal-NF profile 输出
- **THEN** 推荐器 MUST 使用 profile 中的 DataLoader wait、cache read P95、IO-risk 和 GPU step 字段调整 worker、局部性顺序和 AMP 建议
- **AND** 如果没有 profile，推荐器 MUST 给出保守默认和运行 profile 的命令提示

### Requirement: Multimodal-NF cache migration 诊断
训练吞吐 profile 和并行训练推荐器 MUST 能诊断 Multimodal-NF image/LiDAR 派生 cache 的 sidecar schema 状态，并 MUST 在训练尚未进入 GPU step 前给出可执行的 cache 维护建议。

#### Scenario: profile 输出 sidecar schema 统计
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/profile_training_io.py --config <multimodal-nf-config>`
- **THEN** profile 输出 MUST 包含每个启用 image/LiDAR cache 模态的 sidecar schema version 统计
- **AND** 输出 MUST 包含 valid、migration pending、invalid、missing 和 metadata upgrade supported 数量或等价字段
- **AND** 输出 MUST 标明当前 run 是否可能在进入 GPU step 前执行 cache metadata upgrade、rebuild 或 fallback

#### Scenario: 推荐器发现 migration pending
- **WHEN** 用户对 Multimodal-NF image/LiDAR/fusion 配置运行并行训练推荐器，且 cache `.npy` 存在但 sidecar migration pending 数量大于 0
- **THEN** 推荐器 MUST 输出先运行 derived cache 预处理升级的建议
- **AND** 推荐器 MUST NOT 将 `read_only` 作为唯一推荐策略
- **AND** 推荐器 MUST 说明在 migration pending 清零前，首次训练启动可能主要消耗 CPU/磁盘 IO 而不是 GPU

#### Scenario: 推荐器区分缺失与可迁移 cache
- **WHEN** 推荐器检查 Multimodal-NF image/LiDAR cache 状态
- **THEN** 输出 MUST 区分 cache data missing、sidecar migration pending 和 cache invalid
- **AND** 对 migration pending cache MUST 推荐 metadata-only upgrade 或等价预处理命令
- **AND** 对 missing 或 invalid cache MUST 推荐 rebuild/auto 生成或回退策略

#### Scenario: profile 输出训练阶段状态
- **WHEN** profile 或训练启动诊断发现耗时发生在 dataset/cache 构建阶段
- **THEN** 输出 MUST 以机器可读字段标明尚未进入 GPU step 或 loader iteration
- **AND** 输出 MUST 包含 cache validation/migration 耗时摘要，避免用户只能通过 GPU 利用率猜测问题

