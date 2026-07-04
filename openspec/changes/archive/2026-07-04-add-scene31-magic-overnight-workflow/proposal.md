## Why

Scene31 missing-modality 主胜者已经稳定，当前需要在夜间用 GPU 4-7 并行筛选下一批“第二创新点”候选。已有 next-round/BC runner 只能覆盖上一轮候选，且不提供四卡任务队列、MP-DRO 训练策略或本批次独立输出边界。

## What Changes

- 新增 Scene31 magic overnight 本地实验矩阵，覆盖 uniform seed 补齐、failure replay 近似对照、pattern-balanced prototype 训练候选和 MP-DRO 候选。
- 新增 local/manual runner，支持 `overnight_core`、`overnight_all`、`mpfr`、`pbpr`、`mpdro` 分组，按 GPU 列表启动单进程单卡 worker，失败不中断并保留逐 run 日志。
- 扩展 U-MaskBeamJEPA 训练 loss，支持 missing-pattern DRO 的 EMA group loss、softmax group weight、epoch CSV 日志和训练诊断。
- 复用现有 apples-to-apples fresh eval 与 BC summary 口径，输出写入 ignored `outputs/scene31_magic_overnight_lmdb/`，不覆盖旧结果。
- 不新增 package CLI，不改变长期模型 registry surface，不恢复已退役 KD/condBTAPA/beamsoft/weakKD 研究线。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `scene31-next-round-experiment-workflow`: 增加 magic overnight 本地矩阵、四卡 launcher、MP-DRO group 日志和复用 fresh eval/summary 的要求。

## Impact

- 影响 `scripts/` 下 Scene31 本地生成、运行和汇总脚本。
- 影响 `src/kd_sensing/losses/u_mask_beam_jepa.py` 的 opt-in MP-DRO loss 分支。
- 影响 focused tests：Scene31 workflow config/runner smoke、U-MaskBeamJEPA loss focused tests、OpenSpec strict validation。
