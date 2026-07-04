## Why

Scene31 missing-modality 下一轮需要把三条主线与五个 quick screen 放到独立 funnel root 中运行，并把 summary 从单一 `avg_missing` 扩展到缺 1/2/3 个模态的分层口径。上一轮 overnight 已经完成，但其 MPFR/PBPR 多为 proxy，不能继续承载本轮更细的筛选与结论判断。

## What Changes

- 新增 Scene31 funnel local/manual workflow，默认输出到 `outputs/scene31_funnel_lmdb`，不覆盖 next-round、BC、beamsoft weak 或 magic overnight 结果。
- 扩展 apples-to-apples fresh eval / summary 口径，自动建立 pattern 到 missing-count bucket 的映射，并输出 `missing_bucket_mapping.json`。
- 新增 missing-aware checkpoint selection 工具，基于 val split 或 bounded fresh-eval subset 选择 checkpoint，并输出 per-epoch 与 summary CSV。
- 新增 funnel 配置生成、runner 和 summary 脚本，覆盖 JTT 补 seed、MVFR、mild MP-DRO P0/P1 与五个 quick screen run name。
- 对 mild MP-DRO 采用现有 U-MaskBeamJEPA opt-in MP-DRO 分支的最小扩展：支持 `lambda_dro`、full protection 与新版 group 日志列。
- 明确 quick screen 结果只用于晋级判断，不作为论文结论；不恢复 beamsoft、condBTAPA、weakKD 或 AMBER 路线。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `scene31-next-round-experiment-workflow`: 增加 funnel 本地矩阵、missing bucket summary、missing-aware checkpoint selection、runner、summary 与 mild MP-DRO 日志要求。

## Impact

- 影响 `scripts/` 下 Scene31 本地生成、选择、运行与汇总脚本。
- 影响 `scripts/reevaluate_apples_to_apples.py` 与 `scripts/summarize_scene31_bc_next.py` 的 pattern bucket 派生能力。
- 影响 `src/kd_sensing/losses/u_mask_beam_jepa.py` 的 opt-in MP-DRO 权重与日志字段。
- 影响 focused tests：Scene31 workflow summary/generator/runner smoke、U-MaskBeamJEPA MP-DRO focused tests、OpenSpec strict validation。
