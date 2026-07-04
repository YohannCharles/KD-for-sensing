## Context

现有 Scene31 workflow 已有 next-round、BC、beamsoft weak 与 magic overnight 四条本地实验线，均采用 generated YAML + bash runner + apples-to-apples fresh eval + summary 的 local/manual surface。当前 funnel 需要继续复用这些入口，同时补上 missing bucket 分层与 checkpoint selection，避免把 quick screen 单 seed 和 overnight proxy 误当作稳定结论。

## Goals / Non-Goals

**Goals:**
- 默认输出到 `outputs/scene31_funnel_lmdb`，不覆盖已有结果和 checkpoint。
- 复用 `scripts/reevaluate_apples_to_apples.py`、`summarize_scene31_bc_next.py` 与现有 U-MaskBeamJEPA training extension。
- 支持三条主线的 3-seed 配置和五条 quick screen 的 1-seed 配置。
- summary 输出 miss1/miss2/miss3、avg_missing、beam proximity、delta 和 conservative conclusion。
- checkpoint selection 使用 val split 或 bounded subset，不使用 test/fresh eval 目标集调参。

**Non-Goals:**
- 不在本 change 中启动或等待所有长训练完成。
- 不新增 package CLI；funnel 仍是 `scripts/` 下的本地手工 workflow。
- 不恢复 beamsoft、condBTAPA、weakKD 或 AMBER 到本轮 funnel。
- 不实现大规模新模型架构；quick screen 以最小配置/标记和 post-hoc 脚本为主。

## Decisions

1. **bucket 逻辑放在 summary/fresh-eval 共享脚本中。**
   通过 pattern mask 或显式 mapping 计算 `missing_count`，summary 只聚合已有 metrics，不重复模型 forward。这样新旧结果 CSV 都能补分层指标。

2. **funnel runner 复制 magic overnight 的 bash worker 队列形态。**
   现有脚本已经满足单 GPU 单进程、失败不中断和跳过完成 run 的需求；funnel 只替换 run group、输出 root、fresh eval policy 和 summary 调用。

3. **mild MP-DRO 在现有 MP-DRO 分支上最小扩展。**
   原有分支已经维护 EMA loss 与 softmax group weight；本轮只补 `lambda_dro` 混合强度、full protection、`protected_weight` 日志列，避免重写训练循环。

4. **MVFR 先落为可审计配置与 score artifact，占位严格训练接入。**
   严格二阶段 sample replay 需要 dataset sampler 级别改造；本轮先输出 failure score schema 与 MVFR run 配置，并在 runner/summary 中独立标记，后续若结果值得再扩大实现面。

5. **quick screen 只晋级不定论。**
   summary 按阈值标记 `promote_to_full_seeds`、`do_not_promote`、`candidate_second_innovation` 或 `auxiliary_analysis_candidate`，结论文本保持保守。

## Risks / Trade-offs

- [Risk] 历史结果缺少所有 missing patterns，某些 bucket 为空。→ 输出 NaN 和 warning，不中断 summary。
- [Risk] checkpoint selection 如果没有 epoch checkpoints 只能看到 best checkpoint。→ 打印 warning，并仍输出可用 selection summary。
- [Risk] quick screen 配置可能只是 proxy，不等价于最终论文方法。→ manifest tags 和 conclusion 显式标记 quick/proxy。
- [Risk] 全量训练耗时长，无法在实现阶段完成。→ 只验证生成器、summary、selection 和 loss focused tests，训练由用户按 runner 命令启动。
