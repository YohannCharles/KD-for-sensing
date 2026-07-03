## Why

Scene31 night-grid 粗筛已经收敛出少数候选，下一步需要用 seed3/4/5 与 40 epoch 复核稳定性，并小范围验证 uniform sampler 与 selective condBTAPA weak_single 的组合是否能同时保住 full 和提升弱单模态。

## What Changes

- 新增 Scene31 next-round 本地实验配置矩阵，覆盖 es40 seed3/4/5 复核、uniform sampler + condBTAPA weak_single 的 λ=0.05/0.025/0.01 小范围组合，以及少量 P1 配置。
- 新增本地批量 launcher，支持 P0/P1/all 分组、GPU id、skip/overwrite、训练后 fresh eval、失败继续和失败列表输出。
- 新增 next-round fresh eval 汇总脚本，输出 per-run CSV、method mean±std CSV、Markdown 表、相对 proto baseline delta 和阈值 filtered/top10 表。
- 增加最小 sanity check，确认 run name 中的 seed/epoch/λ 与实际配置一致，uniform sampler 和 condBTAPA weak_single 开关正确，missing pattern 与 balanced 公式保持现有口径。
- 不修改 proto baseline、BTAPA tau1 baseline、已有 es20 配置或已有输出目录。

## Capabilities

### New Capabilities
- `scene31-next-round-experiment-workflow`: 记录 Scene31 next-round local/manual 实验配置、launcher、fresh eval 汇总与 sanity check 边界。

### Modified Capabilities
- `experiment-workflow`: 扩展现有 Scene31 night-grid local/manual workflow，增加 next-round 40 epoch follow-up 配置、批量运行和汇总产物要求。

## Impact

- 影响 `configs/scene31/next_round/`、Scene31 本地实验脚本、fresh eval 配置查找路径和 focused tests。
- 运行产物仍限定在 ignored 的 `outputs/scene31_next_round`、`outputs/scene31/` 和 `logs/scene31/next_round`，不提交 checkpoint、日志或 fresh eval 结果。
- 不新增 package CLI、不新增模型结构、不改变已有 baseline 行为。
