## 1. Cache status 与 sidecar-only 迁移

- [x] 1.1 扩展 `multimodal_nf_derived_cache` 的 cache status 结果，区分 valid、migration pending、invalid 和 missing，并暴露 sidecar schema version、待补字段和 validation mode。
- [x] 1.2 实现 metadata-only sidecar upgrade helper：读取旧 sidecar、`.npy` header、cache/source file stat，补齐 v2 lightweight metadata，并使用原子 JSON 写入。
- [x] 1.3 为 upgrade helper 增加安全判定：source path/fingerprint、profile、split、`seq_len`、`num_pred`、shape、dtype、sample_count 不匹配时拒绝迁移。
- [x] 1.4 支持显式 strong validation 路径：需要时重新计算 source fingerprint，记录耗时、结果和 `source_fingerprint_scanned`。
- [x] 1.5 保证 metadata-only upgrade 不读取完整 `.npy` 数据、不重写 `.npy` 文件，并在返回结果中区分 `metadata_upgraded`、`generated`、`rebuilt`。

## 2. Dataset 与预处理集成

- [x] 2.1 将 Multimodal-NF dataset `policy=auto` 接入 migration pending 优先升级路径，升级后重新执行 lightweight status。
- [x] 2.2 保持 `policy=read_only` 不写 sidecar；遇到 migration pending 时输出包含 cache path、source path、原因和预处理命令的清晰错误。
- [x] 2.3 更新 `prewarm_multimodal_nf_derived_cache`，在 `rebuild=false` 时优先执行 sidecar-only upgrade，并汇总 valid/skipped、metadata upgraded、generated/rebuilt、failed、missing 数量。
- [x] 2.4 确保 `rebuild=true` 仍强制重写 `.npy`，且 missing/invalid cache 继续按现有安全策略生成或失败。
- [x] 2.5 扩展 runtime metadata，记录每个 split/模态的 `migration_pending`、`metadata_upgraded`、`cache_generated`、`cache_rebuilt`、validation duration 和 source fingerprint scan 状态。

## 3. Profile 与推荐器诊断

- [x] 3.1 扩展 Multimodal-NF cache 状态统计 helper，供 profile 和推荐器复用，输出 sidecar schema version 分布、valid/migration pending/invalid/missing 计数。
- [x] 3.2 更新 `scripts/profile_training_io.py`，在 Multimodal-NF 配置下输出 cache validation/migration 耗时、是否尚未进入 GPU step 或 loader iteration、以及可能触发 upgrade/rebuild/fallback 的字段。
- [x] 3.3 更新 `scripts/recommend_parallel_training.py`，当发现 migration pending 时推荐先运行 derived cache 预处理升级，不把 `read_only` 作为唯一建议。
- [x] 3.4 更新推荐输出说明，区分 cache data missing、sidecar migration pending 和 cache invalid，并给出对应的 `conda run -n kd_mm_beam python scripts/preprocess.py --config configs/preprocess/multimodal_nf_derived_cache.yaml ...` 命令建议。

## 4. 测试覆盖

- [x] 4.1 在 Multimodal-NF fixture 中构造 v1 sidecar，验证 `policy=auto` 执行 metadata-only upgrade 且 `.npy` mtime 或内容不变。
- [x] 4.2 验证 `policy=read_only` 遇到可迁移 v1 sidecar 时失败、sidecar 不被改写、错误信息包含 migration pending 和预处理命令。
- [x] 4.3 验证 source/profile/split/window/shape/dtype/fingerprint 不匹配的旧 sidecar 拒绝 metadata-only upgrade，并按 policy 清晰失败或安全重建/回退。
- [x] 4.4 验证 strong validation 迁移会扫描 source fingerprint，fingerprint 不匹配时拒绝升级。
- [x] 4.5 验证预处理输出包含 valid/skipped、metadata upgraded、generated/rebuilt、failed、missing 汇总字段。
- [x] 4.6 验证 profile 和推荐器输出 sidecar schema、migration pending、阶段诊断和预处理建议字段。

## 5. 验证与本地运行建议

- [x] 5.1 运行 `conda run -n kd_mm_beam pytest tests/test_multimodal_nf_dataset.py -q`。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_parallel_training_recommendations.py -q`。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_epoch_subsampling.py -q`，确认 locality/subsampling 既有行为未受影响。
- [x] 5.4 运行 `openspec validate fix-multimodal-nf-cache-sidecar-migration --strict`。
- [x] 5.5 在真实本地数据上运行 train/test cache sidecar 预处理升级命令，并确认输出不纳入源码变更。
- [x] 5.6 用 `fusion_all_tasks` 配置做一次小样本启动验证，建议命令包含 `training.epoch_subsampling.enabled=true`、`training.epoch_subsampling.fraction=0.1`、`training.epoch_subsampling.order=locality` 和 `output.progress.enabled=false`。
