## 1. 吞吐诊断与推荐

- [x] 1.1 扩展 `scripts/profile_training_io.py`，记录 MMW HiST-Beam 的 enabled modalities、seq_len、batch size、worker/prefetch/pin_memory、分模态 getitem、DataLoader wait、step 分解和 IO-risk 字段。
- [x] 1.2 为 MMW image-heavy profile 增加 focused 测试，使用 `conda run -n kd_mm_beam pytest ... -q` 验证字段稳定且能标记 loader wait 支配 step。
- [x] 1.3 扩展 `scripts/recommend_parallel_training.py` 和推荐引擎，使其根据 image modality、seq_len、batch size、parallel runs、worker 数和 profile/OOM 信号输出 memory-aware 覆盖建议。
- [x] 1.4 为推荐器增加 focused 测试，验证 image-heavy MMW 配置不会只推荐 AMP，并会优先限制 worker、batch size 或并行度。

## 2. MMW RGB/ImageNet Image 派生缓存

- [x] 2.1 设计并实现 image-derived cache metadata schema，包含源 image fingerprint、image size、image profile、transform version、dtype、shape 和生成时间。
- [x] 2.2 扩展 cache policy 解析，支持 `data.cache.image.policy` 的 `off`、`read_only`、`auto`、`rebuild`，并继续拒绝所有旧 `image_motion_*` 字段。
- [x] 2.3 在 RGB/ImageNet image 加载路径中接入 image-derived cache，确保 cache hit 与原始 PNG 解码路径返回等价 tensor。
- [x] 2.4 增加 image-derived cache 预热入口或预处理配置，输出扫描数、生成数、跳过数、失败数、coverage 和 cache 总大小。
- [x] 2.5 增加 focused tests，使用 `conda run -n kd_mm_beam pytest ... -q` 验证 cache hit/miss、read_only 不写入、未启用 image 不访问 cache、以及旧 image motion 字段仍被拒绝。

## 3. MMW Dataset 内存边界

- [x] 3.1 将 GPS/mmWave normalizer 拟合改为 streaming 或可释放临时统计，避免 dataset 初始化后保留所有样本的 per-sample sequence cache。
- [x] 3.2 在 dataset/runtime metadata 中记录 scaler 来源、拟合样本数、是否 streaming、cache policy 和 worker 内存相关配置。
- [x] 3.3 调整 MMW dataset 的 image/GPS/mmWave 按需加载测试，覆盖 `modalities=[gps,mmwave]` 时不读取 image、不初始化 image cache。
- [x] 3.4 增加多 worker DataLoader focused 测试或 profile 断言，确认初始化阶段不会复制全量 GPS/mmWave per-sample 大数组。

## 4. LOSO Stage 数据构建边界

- [x] 4.1 将 LOSO 数据构建拆分为 source train、target adapt、target test 的 stage-local helper，source stage 只构建 source loader。
- [x] 4.2 调整 `hist_beam_loso_execution`，在每个 stage 内按需构建 loader，并在 stage 结束、失败或中断时关闭对应 DataLoader worker。
- [x] 4.3 确保 scaler/normalizer 等跨 split 轻量状态通过 artifact 或 metadata 显式传递，不依赖提前构建 target dataset。
- [x] 4.4 增加 focused tests，验证 `source_train` 不构建 target adapt/test dataset，target stages 会延迟构建所需 loader。

## 5. Source Prototype 按需生成

- [x] 5.1 增加 prototype strategy 配置和 executor 决策逻辑，使 source-only baseline 默认跳过 prototype 生成并记录 skipped reason。
- [x] 5.2 在需要 prototype 的 variant 前生成或复用匹配 fold、source scenes、variant、seed 和 proto_type 的 source prototype artifact。
- [x] 5.3 为 prototype generation pass 增加 progress callback，记录 phase、processed batches、processed samples、duration 和 coverage。
- [x] 5.4 更新 LOSO summary/metrics，区分 source training duration 与 prototype generation duration，并记录 prototype status。
- [x] 5.5 增加 focused tests，验证 `v0_flat` 跳过 prototype、`v5_adapter_proto` 按需生成或复用、缺失 artifact 有清晰 no-op 或 failure 诊断。

## 6. 集成验证与文档

- [x] 6.1 运行 OpenSpec 校验：`openspec validate optimize-mmw-hist-beam-throughput --strict`。
- [x] 6.2 运行相关 focused tests：`conda run -n kd_mm_beam pytest <focused tests> -q`。
- [x] 6.3 运行架构边界和 CLI 快速检查：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`、`conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`、`conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`。
- [x] 6.4 使用小样本 MMW/HiST-Beam profile 验证 image cache、stage-local loader、prototype strategy 和 recommender 输出，不提交生成的 cache、日志或 checkpoint。
- [x] 6.5 更新 README 或 docs 中 MMW HiST-Beam 运行建议，说明 image-derived cache、并行训练保守覆盖、profile 命令和 OOM 诊断路径。
