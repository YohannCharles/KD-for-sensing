## Context

当前 Scene31 三 seed 复评显示 ordinary prototype baseline 在 full、avg_missing、missing_gps、missing_radar 上更稳，BTAPA tau1 主要提升 radar_only / lidar_only。用户本轮目标是今晚能用 8 张 A40 跑 60+ 个短实验，优先筛出不伤主指标的弱单模态改进方向。

已有项目约束要求复用当前 `src/kd_sensing` 包结构、当前训练入口、统一 missing pattern helper 和 checkpoint resolver；本变更不得恢复旧 KD、复杂 JEPA、RBMA 扩展或根目录训练入口。

## Goals / Non-Goals

**Goals:**

- 以 ordinary proto es20 base 为统一基础，自动生成 A-F 和 baseline/reference 共 64 个配置与 manifest。
- 让 pattern-conditional BTAPA、sampler、CE reweight 和 mask adapter 能先跑 A-D 类粗筛。
- 提供单进程单 GPU 的 8 卡 launcher、fresh eval、analysis 和 summary 链路。
- E/F 仅作为 opt-in weak-pattern KD / latent prediction probe，默认不影响主线。

**Non-Goals:**

- 不继续扩展 RBMA attention 主体。
- 不重新引入完整 JEPA 或原始模态预测。
- 不把 KD、fullaux 或全局 BTAPA 写成主方法。
- 不覆盖已有 proto / BTAPA tau1 outputs、logs、checkpoint 或 eval。

## Decisions

1. pattern-conditional BTAPA 在 prototype loss 内 sample-wise 混合 target。
   - 选择：同一个 batch 内按 pattern mask 生成 ordinary one-hot/proto target 与 BTAPA soft target，再逐样本混合 loss。
   - 原因：用户明确要求 sample-wise，且 full batch 判断会在 mixed missing pattern sampler 下错误。
   - 替代：为 BTAPA pattern 单独拆 batch；会增加训练循环复杂度和 dataloader 开销。

2. missing pattern sampler 作为训练 mask 生成 helper，而不是新 dataset。
   - 选择：复用统一 missing pattern API，在训练 batch 准备/extension mask 路径中根据 config 采样 `[B, M]` availability mask。
   - 原因：改变 pattern 频率不应修改 dataset split 或真实输入。
   - 替代：WeightedRandomSampler 按样本重采样；当前目标是 pattern 频率，不是数据样本频率。

3. hard pattern reweight 只默认加权 CE。
   - 选择：训练 objective loss 使用 sample-wise CE，再按 pattern weight 平均；proto loss 默认不乘 pattern weight。
   - 原因：用户目标是加强 hard pattern 分类监督，同时保持 ordinary proto 主线稳定。

4. mask adapter 只插在 fusion 后 beam head 前。
   - 选择：新增小 MLP 输出 gamma/beta，`h * (1 + gamma * scale) + beta * scale`。
   - 原因：最小改动，不改 fusion 主体，不影响未启用配置。

5. night grid 脚本保持文件级入口，不新增 console script。
   - 选择：`scripts/*.py` / `.sh` 只做本地 orchestration 和分析。
   - 原因：这是本地实验网格，不是长期包 API。

## Risks / Trade-offs

- [Risk] 训练 runtime 现有 loss 可能不是 per-sample CE。→ 先用已有 forward/logits 路径补最小 per-sample 分支，并保留未启用配置的旧行为。
- [Risk] E/F full-branch teacher 会增加显存和时间。→ 默认关闭，只在指定 pattern 上计算，eval 不启用。
- [Risk] 64 个配置由脚本生成，容易与手写基线字段漂移。→ 统一 base template，manifest 作为唯一调度清单。
- [Risk] launcher 对 completed 的判断不可能覆盖所有历史命名。→ 复用 summary/checkpoint resolver，缺失时 warning 并记录 failed/completed 清单。

## Migration Plan

1. 新建 base template 与生成脚本，dry-run 确认 64 个 manifest 行。
2. 先验证 A-D 相关 runtime smoke，不启动完整训练。
3. 运行 launcher `--dry_run` 确认 GPU 绑定与过滤。
4. 训练完成后用 fresh eval 和 analyze 脚本统一比较。
5. 若候选方向成立，再另开 follow-up 做 seed3 / 40 epoch，不把本轮粗筛结论写成最终 claim。
