## Context

MMW Town10 HiST-Beam LOSO 多模态训练当前主要卡在 CPU 数据管道，而不是 CUDA 计算。`seq_len=8`、`image+gps+mmwave`、两个 source scene 拼接和较大的 batch 会让一个 batch 触发数百张 PNG 的读取、解码、resize 和张量化；DataLoader 多 worker 又会把 dataset 内部缓存、CSV 派生数据和 Python 对象复制到多个进程，导致系统内存高、GPU 显存低、GPU 等待数据。

当前实现还有两个放大因素：

- LOSO `source_train` 构建 dataloaders 时同时构建 source、target adapt 和 target test，source stage 尚未使用 target 数据却提前占用内存。
- source training 完成后会无条件再扫一遍 source train split 生成 prototype，即使 `v0_flat` 等 source-only baseline 不需要 prototype，也会再次经过 image-heavy 数据加载路径。

本变更的长期目标是把临时降载参数升级为可配置、可诊断、可回归的机制，同时保持现有 MMW 样本契约、HiST-Beam 指标和 LOSO 结论语义不变。

## Goals / Non-Goals

**Goals:**

- 为 MMW RGB/ImageNet image 序列提供受控派生缓存，减少重复 PNG 解码和 resize。
- 让 LOSO stage 只构建当前 stage 必需的数据集和 DataLoader，降低 source training 常驻内存。
- 让 source prototype 生成按 variant 和配置启用，并提供进度与 no-op 诊断。
- 让 throughput profile 和 parallel recommendation 能识别 MMW image-heavy 场景的 loader wait、worker RSS 和 OOM 风险。
- 避免 GPS/mmWave normalizer 拟合在 dataset 初始化阶段无界缓存所有样本特征。
- 保持 `input_beam`、`target_beam`、image、GPS、mmWave tensor shape 和现有 metric 语义兼容。

**Non-Goals:**

- 不恢复 image motion mask cache，也不新增旧 `image_motion_*` 配置兼容路径。
- 不改变 HiST-Beam 模型结构、loss 数学定义或 LOSO 实验矩阵的科学问题。
- 不把本地缓存、预热产物、checkpoint、日志或训练输出纳入源码。
- 不默认要求用户必须先离线缓存才能训练；无缓存时仍可在线读取，只是会有性能提示。

## Decisions

1. **新增 MMW RGB/ImageNet 派生缓存，而不是复用旧 image motion cache。**

   新缓存以 RGB/ImageNet 模型输入为契约，保存 resize/normalize 后的 tensor 或等价数组，cache key 至少包含原图路径、mtime/size 或内容 fingerprint、image size、profile 和 transform version。它只服务启用 image modality 的 MMW/DeepSense6G 图像序列，不解析任何旧 motion mask 字段。

   备选方案是恢复 image motion cache，但它与当前 RGB/ImageNet 输入语义不同，会重新引入已删除的兼容面，且无法直接解决当前 PNG decode/resize 重复开销。

2. **cache policy 扩展为受支持 image-derived cache，但保持显式模态边界。**

   `data.cache.policy` 继续统一控制 cache 行为；新增 `data.cache.image.policy` 只允许指向 RGB/ImageNet 派生缓存。未启用 image 时不得检查该 cache；启用 image 且 policy 为 `auto` 时可按需生成；`read_only` 缺失时在线计算但不写入。

   备选方案是在 dataset 内隐式写缓存，但这样不可诊断，也不符合当前 cache policy 可追踪契约。

3. **LOSO dataloaders 改为 stage-local 构建。**

   `source_train` 只构建 source train loader；`source_only_target_test_eval` 和 `adapted_target_test_eval` 才构建 target test loader；`target_adaptation` 才构建 target adapt loader。需要共享的 normalizer/scaler 通过轻量 artifact 或 metadata 显式传递，不通过提前常驻 target dataset 实例实现。

   备选方案是保持一次性构建全部 loader，但这会继续让 source stage 持有暂不用的 target 数据结构和 worker。

4. **source prototype 生成变为按需策略。**

   `v5_adapter_proto`、`v6_radio_proto`、`adapter_radio_proto` 等后续确实需要 prototype 的 run 才生成并缓存 source prototype；`v0_flat`、`v1_hierarchical` 等 source-only baseline 默认跳过。若用户显式要求保存 prototype，系统仍可生成，但必须记录原因、耗时和进度。

   备选方案是所有 source training 后继续生成 prototype，优点是 artifact 一致，缺点是 image-heavy 场景下代价过高且对 source-only baseline 无科学价值。

5. **normalizer 拟合采用 streaming/轻量统计，不保留每个样本大数组缓存。**

   GPS/mmWave scaler 拟合时可以复用 frame-level 小缓存或 streaming stats，但完成拟合后必须释放 per-sample sequence cache，且 DataLoader worker 中不得复制初始化阶段产生的大数组缓存。运行 metadata 记录 normalizer 来源、样本数和是否使用 streaming。

   备选方案是完全关闭 normalization；短期可作为覆盖参数，但长期不应牺牲实验默认语义。

6. **profile 和 recommender 以“数据等待支配 step”为第一诊断信号。**

   profile 必须记录 image/GPS/mmWave 分模态 getitem、DataLoader wait、GPU step、worker 数、batch size、seq_len、并行 run 数和 worker RSS 风险。推荐器在 MMW image-heavy 场景下优先限制并行 run、worker 和 batch，再建议 AMP；AMP 只能缓解模型算力和显存，不解决 PNG 解码。

## Risks / Trade-offs

- [Risk] 派生 image cache 占用磁盘空间。→ Mitigation：cache metadata 记录总大小、样本覆盖率和清理提示；默认不提交缓存。
- [Risk] cache stale 导致读取旧图像特征。→ Mitigation：cache key 包含 transform version 和源文件 fingerprint；不匹配时按 policy rebuild 或在线计算。
- [Risk] stage-local loader 改动影响 scaler 复用。→ Mitigation：将 scaler 作为轻量 artifact/metadata 从 source train 明确传递，增加 focused tests 覆盖 train/test shape 与 normalization 一致性。
- [Risk] prototype 按需生成可能让后续 adaptation 找不到 artifact。→ Mitigation：执行器在需要 prototype 的 variant 前检查并生成或复用，缺失时给出清晰失败或 no-op 诊断。
- [Risk] 推荐参数过保守导致单实验吞吐下降。→ Mitigation：区分后台并行推荐与单实验 profile 推荐；输出理由和可调范围。

## Migration Plan

1. 增加配置解析、cache policy 和 metadata 字段，但默认保持无缓存也可运行。
2. 添加 profile/recommender 对 MMW image-heavy 的诊断字段，先用于验证当前瓶颈。
3. 实现 image-derived cache 和预热入口，提供 read-only/auto/rebuild 行为。
4. 改造 LOSO stage-local loader 和 prototype 按需生成，保持原 summary schema 兼容。
5. 增加 focused tests 和 smoke profile，验证缓存等价、stage 构建边界、prototype 跳过和 recommender 输出。
6. 用小样本 MMW 或 fixture 运行回归；全量数据只作为本地性能验证，不进入源码。

## Open Questions

- image-derived cache 首版保存 processed tensor 还是 frozen image encoder feature？建议首版保存 processed tensor，后续可扩展 feature cache。
- cache fingerprint 是否需要内容 hash？建议默认使用 path、size、mtime 和 transform version，必要时支持 strict hash。
- MMW full matrix 是否应默认串行 target scene、并行 seed，还是由推荐器根据 RAM/GPU 数动态生成命令？建议先由推荐器输出，不自动改实验矩阵。
