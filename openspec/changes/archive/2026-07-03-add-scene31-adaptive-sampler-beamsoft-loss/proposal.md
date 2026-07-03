## Why

Scene31 night-grid P0 fresh eval 显示 `proto_sampler_uniform_es40` 是当前 missing-modality 主胜者，但它仍使用固定 uniform pattern exposure 和普通 hard-label CE。下一轮实验需要在不改变既有 baseline/已完成 run 行为的前提下，验证两个低侵入方向：按 pattern 困难度动态调整 exposure，以及利用 beam index 邻域结构训练。

## What Changes

- 新增 opt-in adaptive pattern-balanced sampler，在 U-MaskBeamJEPA 缺失模态 mask 采样处基于 EMA loss/gap 动态调整 pattern probability，并写出 `adaptive_sampler_log.csv`。
- 新增 opt-in `beam_neighborhood_ce` 与 `label_smoothing_ce`，默认仍保持现有 CE/focal 行为；beam-neighborhood loss 使用 circular Gaussian soft target 与 hard CE 混合。
- 扩展 Scene31 next-round generator/manifest，新增 B、C、B+C 和 label smoothing run names，均基于 `proto_sampler_uniform_es40` 主线且不启用 condBTAPA/weakKD。
- 新增 `scripts/run_scene31_bc_next.sh`，支持按 group 训练/复评/汇总 BC 实验，并在实验矩阵中包含需要训练的 `amr_net_supervised` 与 `amber_full_architecture` baseline。
- 扩展 fresh-eval summary，使 method-level 汇总按 seed 归并，主排序为 `avg_missing`、`full`、`overall_mean`、`balanced`，并输出 delta vs proto 与 delta vs current uniform winner。
- 不删除、不覆盖既有 checkpoint、P0 输出或旧 P0 generator 行为；已有配置未 opt-in 时数值路径不变。

## Capabilities

### New Capabilities

- `adaptive-pattern-balanced-sampler`: 覆盖 adaptive sampler 的配置、EMA 更新、概率裁剪、warmup、fallback、日志和 sanity check 契约。

### Modified Capabilities

- `scene31-next-round-experiment-workflow`: 扩展 Scene31 local/manual next-round 矩阵、BC launcher、baseline inclusion、fresh-eval summary 和 delta vs uniform winner 契约。
- `soft-beam-label-training`: 增加 beam-neighborhood hard/soft CE 混合 loss 与 label smoothing 对照的配置和训练契约。

## Impact

- 主要影响 `src/kd_sensing/losses/beam.py`、`src/kd_sensing/losses/u_mask_beam_jepa.py`、Scene31 config generator/manifest、BC launcher、summary 脚本和 focused tests。
- Python 验证命令必须使用 `conda run -n kd_mm_beam ...`；OpenSpec 变更需运行 `openspec validate add-scene31-adaptive-sampler-beamsoft-loss --strict`。
- 本变更不新增依赖，不改变默认训练入口，不读取真实 `dataset/`，也不把训练输出、logs、cache 或 checkpoint 纳入源码变更。
