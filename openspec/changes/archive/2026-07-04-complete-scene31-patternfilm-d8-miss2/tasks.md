## 1. PatternFiLM d8 配置与实现确认

- [x] 1.1 检查 seed1 配置、sampler、epoch、checkpoint policy 和禁用方法线状态，记录不一致项。
- [x] 1.2 补齐 `model.primary.pattern_film` 的实际消费路径，确保 d8/pre_head/init_identity 语义生效且只基于 availability mask 条件化。
- [x] 1.3 扩展 funnel generator/manifest，使 PatternFiLM d8 生成 seed1-5，seed2-5 与 seed1 只差 seed。

## 2. Fresh Eval 与 Missing Bucket

- [x] 2.1 扩展 apples-to-apples fresh eval 默认 patterns，纳入支持模态集合下的 miss2 patterns，且不使用 `--max-batches`。
- [x] 2.2 扩展 `missing_bucket_mapping.json`，为每个 pattern 写出 `available_modalities`、`missing_modalities` 和 `missing_count`。
- [x] 2.3 新增 `scripts/run_scene31_patternfilm_d8.sh`，支持 train/eval group、`--extra-root`、多 GPU 队列、默认跳过与 `--overwrite-eval`。

## 3. Summary 与 Sanity

- [x] 3.1 新增 `scripts/summarize_scene31_patternfilm_d8.py`，输出 per-run、method mean/std、delta、rank、bucket mapping 和保守 conclusion。
- [x] 3.2 加入 PatternFiLM d8、miss2 pattern、fresh eval status 和 metric sanity checks。
- [x] 3.3 运行 `openspec validate complete-scene31-patternfilm-d8-miss2 --strict`、相关 `conda run -n kd_mm_beam pytest ...` 和脚本 smoke。

## 4. 本地运行

- [x] 4.1 使用 `conda run -n kd_mm_beam` 生成/校验 Scene31 funnel configs。
- [x] 4.2 按用户命令补跑 PatternFiLM d8 seeds 2-5；若训练耗时或资源阻塞，记录启动状态与剩余命令。
- [x] 4.3 用新增 miss2 patterns 重跑 PatternFiLM d8 与 uniform fresh eval，并生成最终 summary。
