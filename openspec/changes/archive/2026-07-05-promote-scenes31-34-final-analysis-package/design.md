## Context

当前 Scene31-34 主结果已经完成：core prototype n=5、classifier baseline n=3、AMR/AMBER-lite external-lite seed1，且 `mask_suspect=false`。本轮任务只读取既有 fresh eval、summary、profile 和训练采样日志，不启动训练、不覆盖 checkpoint、不新增复杂模型。

## Goals

- 让主方法相对 Bernoulli、classifier、natural/uniform 和 AMBER-lite 的提升具备统计证据。
- 证明提升不是少数 pattern 平均造成，而是在 pattern-level 和 error CDF 上更稳定。
- 说明 random non-empty subset exposure 与 Bernoulli randomdrop 的采样分布差异。
- 统一最终论文表格、final notes 和 final conclusion。
- 提供一键 final analysis runner，便于复现最终分析产物。

## Non-Goals

- 不训练新模型。
- 不补 AMR/AMBER seed2/3。
- 不继续 reliability fusion、PatternFiLM、JTT、MVFR、MPDRO、beamsoft、condBTAPA、weakKD 或新融合模块。
- 不把 n=1 external baseline 当成 seed-level 显著性结果。

## Decisions

### 1. 统计脚本优先使用 per-sample predictions

`significance_tests.py` 优先读取 `fresh_eval_with_scene` / `fresh_eval_maskfix_with_scene` 的 `predictions_by_pattern.csv` 与 `pattern_metrics.csv`。如果 fixture 或历史 artifact 缺少 predictions，则降级读取 summary CSV，并写 warning；降级路径只提供弱统计。

### 2. Bootstrap over samples/patterns 保持 artifact-only

Bootstrap 在已有 prediction rows 上做 paired delta，不重新运行 eval。accuracy/within@3 的 delta 为 method - baseline；MAE、MAE@75 和 Top1 drop 的 delta 为 baseline - method，因此正数始终表示主方法更好。

### 3. Compute profile 不启动训练

Profile 从 checkpoint、startup summary、train log 和 timing CSV 读取参数量、模型大小、训练时间和显存；没有已记录的 eval latency 时输出 NaN 与 notes，不启动正式 fresh eval 或训练。

### 4. Final table updater 只整合已有产物

`update_final_paper_tables.py` 复用已有 main paper table exporter，并附加 significance、pattern win-count、sampling distribution 和 final notes。缺失输入时生成空表或 notes，而不是失败整个 workflow。

### 5. Final polish 统一统计和推理成本口径

`significance_summary.csv` MUST 分开 seed mean delta 与 per-sample bootstrap mean/CI，accuracy-like 指标内部保留 fraction，同时输出 percentage points；MAE 保留 beam index 单位。Sanity check 只检验 bootstrap mean 是否落在同一 bootstrap CI 内，避免把 seed mean delta 与 per-sample CI 混读。

Inference latency benchmark 只加载已有 best checkpoint 与固定 eval dataloader，计时单次 forward warmup/benchmark batches，不运行 missing eval 全矩阵、不改写正式 eval 指标。加载失败时输出 NaN 与 warning，final compute table 继续生成。

Paper plot variants 作为新增输出文件生成，不替换原始分析数据；正文优先使用 Top1 degradation paper 图、delta-vs-Bernoulli pattern 图和 paper CDF 图。

## Validation

- `openspec validate promote-scenes31-34-main-missing-count --strict`
- `conda run -n kd_mm_beam pytest tests/test_scene31_34_final_analysis.py -q`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
