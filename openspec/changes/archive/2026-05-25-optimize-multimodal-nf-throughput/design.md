## Context

当前 Multimodal-NF dataset 使用 worker-local HDF5 handle 按样本懒加载启用模态。GPS 每个历史窗口只读取小型位置数组；image 读取 gzip 压缩的 `224x224x3` RGB 序列；LiDAR 读取 gzip 压缩的 `10000x3` 点云序列。实测 `profile_getitem_components()` 显示 GPS 约 0.0005 秒/样本，而 image 约 0.11 秒/样本、LiDAR 约 0.04 秒/样本，fusion 的 dataset 侧耗时主要由 image/LiDAR 组成。

现有配置已经支持 DataLoader worker、pin memory、prefetch、non-blocking transfer、AMP 和 profile 入口，但 Multimodal-NF image/LiDAR 还缺少专用派生数据路径，导致每个 epoch 反复解压原始 HDF5 大 chunk。该变更需要在不改变样本字段、target 语义和模型接口的前提下，提高可诊断性和可复用训练吞吐。

## Goals / Non-Goals

**Goals:**

- 为 Multimodal-NF image/LiDAR 提供显式派生缓存或重打包数据契约，减少训练时重复 gzip 解压和大数组转换。
- 保持按模态懒加载：未启用 image/LiDAR 时不得访问对应派生缓存或原始大数组。
- 在 runtime metadata 和 profile 输出中记录真实数据来源、缓存策略、命中/生成状态和模态级耗时。
- 给含 image/LiDAR 的 Multimodal-NF 配置提供安全默认建议，包括 AMP、worker、prefetch、test worker 和 progress 降噪。
- 用 fixture 和 focused profile 测试验证行为，不依赖提交真实全量数据或生成大产物。

**Non-Goals:**

- 不改变 Multimodal-NF 任务定义、codebook flatten 规则、beam/LOS/link target 字段或指标口径。
- 不引入新的模型结构优化作为本 change 的必需内容。
- 不把派生缓存、重打包数据、profile 输出、checkpoint 或训练日志纳入源码管理。
- 不默认删除或改写用户已有真实 HDF5 数据。

## Decisions

### 1. 采用可选派生缓存层，而不是直接替换原始 HDF5

Multimodal-NF dataset 增加 image/LiDAR 派生数据解析逻辑：优先按配置选择派生缓存或原始 HDF5。派生数据可以是按 city/split 切分的 HDF5/NPZ/Zarr/内存映射文件，具体格式由实现选择，但必须能按 `history_indices` 返回与原始 adapter 等价的 tensor。

理由：原始数据是本地输入资料，项目规则不允许自动迁移、删除或改写真实数据。派生缓存作为 ignored 本地产物更适合迭代，也便于失败后回退到原始 HDF5。

备选方案：训练时增大 HDF5 rdcc cache 或 DataLoader worker。该方式成本低但不能消除 image/LiDAR gzip 大 chunk 的重复解压，只适合作为短期配置建议。

### 2. 配置策略显式化

为 Multimodal-NF 增加模态级派生缓存策略，建议语义如下：

- `off`: 只读原始 HDF5。
- `auto`: 有派生缓存则读取，缺失时允许生成。
- `read_only`: 只读派生缓存，缺失时报错。
- `rebuild`: 重新生成派生缓存后读取。

策略字段应位于 `data.dataset` 或 `data.cache.multimodal_nf` 下，并在解析后进入 dataset runtime metadata。实现时可以先支持 image/LiDAR 子集，GPS 保持原始 HDF5 读取。

理由：用户需要清楚知道训练是否在生成缓存、是否复用缓存，以及缓存缺失时为何变慢或失败。

### 3. 派生缓存生成与训练读取解耦

实现应提供预处理入口或可复用 helper 生成派生缓存；`auto` 可在 dataset 初始化或首次访问时触发小范围生成，但必须限制输出目录并记录 metadata。大规模缓存预热更适合显式预处理命令。

理由：把全量生成放进训练首个 batch 会造成不可解释的长时间卡顿。显式预热能配合 tmux/后台训练和 profile 复现。

### 4. 保留 profile 作为验收入口

`scripts/profile_training_io.py` 应继续作为统一诊断入口，并补充 Multimodal-NF 需要的结构化字段：模态级耗时、数据来源、派生缓存策略、缓存命中/缺失/生成统计、train/test worker 参数、AMP 状态、samples/s 和 CUDA memory。

理由：该问题本质是数据路径和模型 step 共同造成的吞吐问题，单看 epoch 总时长无法定位。profile 输出必须能回答“慢在 image 解压、LiDAR 解压、DataLoader wait、transfer 还是 forward/backward”。

### 5. 配置优化作为建议而非强制迁移

含 image/LiDAR 的 Multimodal-NF 示例配置可以启用更适合 CUDA 训练的默认建议：`pin_memory=true`、`persistent_workers=true`、合理 `num_workers`/`prefetch_factor`、验证 worker 不为 0、`training.amp.enabled=true`、后台 progress 降噪。已有配置仍可通过显式覆盖回到旧行为。

理由：不同机器 CPU/GPU/磁盘差异明显，强行固定 worker 或 batch size 容易在小机器上适得其反。

## Risks / Trade-offs

- 派生缓存占用磁盘空间 → metadata 记录缓存路径、格式、样本数和原始文件 fingerprint；文档提示可删除缓存并回退到 `off`。
- 缓存与原始 HDF5 不一致 → 缓存 sidecar 必须记录原始 path/fingerprint、profile、seq_len、num_pred、shape/dtype；不匹配时拒绝读取或触发 rebuild。
- `auto` 首次运行生成缓存导致启动慢 → profile 和 runtime metadata 明确标记生成耗时；推荐全量训练前先运行预热命令。
- 多 worker 同时生成同一缓存可能竞争 → 生成逻辑需要文件锁、原子写入临时文件后 rename，或要求只有主进程生成。
- 派生格式选择过早固化 → spec 只约束行为与 metadata，不强制具体格式；实现可先选最小可维护格式。
- AMP 可能改变数值细节 → AMP 作为配置建议和可选默认，不改变 CPU/FP32 回退路径；测试只检查训练路径可运行和 metadata 正确。

## Migration Plan

1. 先扩展 profile 和 metadata，建立基线：GPS/image/LiDAR/fusion 的模态级耗时可被稳定记录。
2. 实现 image/LiDAR 派生缓存 helper 和 dataset 读取分支，默认策略保持 `off` 或兼容现状。
3. 增加预处理入口或配置驱动缓存预热命令，并用小型 fixture 验证缓存生成、读取和失效。
4. 更新 Multimodal-NF fusion 示例配置与推荐器输出，给出含 image/LiDAR 的吞吐建议。
5. 用 profile 对比原始 HDF5 与派生缓存路径，记录 samples/s、模态 `__getitem__` 和 DataLoader wait 改善。

回退策略：将派生缓存策略改为 `off`，删除 ignored 缓存目录，不影响原始 HDF5 和既有训练配置。

## Open Questions

- 首个实现应选择 HDF5 contiguous dataset、NPZ shard、Zarr 还是 memory-mapped `.npy` 作为派生格式？建议优先选择依赖最少、fixture 易测、随机窗口读取性能稳定的格式。
- image 派生缓存是否保存 uint8 channel-first，还是预转换为 float32 `[0,1]`？uint8 更省磁盘，float32 可减少训练时转换。
- LiDAR 派生缓存是否保存原始 10000 点，还是增加可选降采样/特征化路径？本 change 建议先保持原始点云语义，避免改变模型输入含义。
