## Context

现有 Scene31 next-round/night-grid workflow 已经提供 manifest-backed 配置生成、串行或单 CUDA_VISIBLE_DEVICES runner、apples-to-apples fresh eval 和 summary。今晚目标是在不大改训练主干的前提下，用 GPU 4-7 跑下一批候选，并把结果写到独立 root，便于明天继续汇总与复盘。

## Goals / Non-Goals

**Goals:**

- 提供一个本地 magic overnight 矩阵与 manifest，默认输出到 `outputs/scene31_magic_overnight_lmdb`。
- 提供真正的单进程单 GPU worker 队列，支持 4 张 GPU 并行训练，失败不中断，日志按 run 分离。
- 对 MP-DRO 提供 opt-in 训练扩展：按 missing pattern 维护 EMA loss，用 softmax/tau 得到 group weight，并写出 `mpdro_group_log.csv`。
- 复用现有 fresh eval、checkpoint resolver、summary 脚本和 `kd-sensing-train` 入口。

**Non-Goals:**

- 不新增 package CLI，不把本地 runner 升级为长期公开入口。
- 不在今晚实现复杂二阶段 dataset-level sample replay 或严格 prototype post-training re-centering；这批配置会在 run metadata 中标记为 overnight proxy/minimal 版本。
- 不恢复 condBTAPA、beamsoft、weakKD 等已确认不主推路线。

## Decisions

1. **新增独立 generator，而不是改 next-round generator。**
   Magic overnight 的命名、输出 root 和候选集合不同，独立 `generate_scene31_magic_overnight.py` 可以避免污染 `configs/scene31/next_round/` 的既有 manifest。

2. **runner 使用 bash worker 队列。**
   训练任务本身仍是单进程单卡，runner 将 run name 写入共享队列文件，每个 GPU 一个 worker 顺序取任务。这样无需 DDP 或新增依赖，也能保证每张卡同一时间只跑一个训练任务。

3. **MP-DRO 落在 U-MaskBeamJEPA training extension。**
   该 extension 已经知道 batch 的 missing pattern、logits 和 labels，可以最小侵入地计算 per-pattern CE、EMA loss、softmax group weights，并通过现有 sample weight 机制影响 supervised CE。

4. **MPFR/PBPR 先用可运行 proxy 配置。**
   JTT-style sample replay 和 strict prototype re-centering 需要额外失败样本缓存、dataset replay 或 checkpoint postprocess，今晚实现风险高。当前矩阵保留原始/魔改命名，但通过 manifest tags 标明 `overnight_proxy`，具体采用 pattern-aware replay weights 与 pattern-balanced prototype training。明天若结果有希望，再补严格二阶段实现。

## Risks / Trade-offs

- [Risk] MPFR/PBPR proxy 与附件中的完整算法不完全等价。→ Mitigation：manifest tags 和最终说明明确标记，避免论文或报告误用。
- [Risk] 后台 overnight run 时间长，无法在启动前跑全量回归。→ Mitigation：先跑 generator/config/load focused checks 和 OpenSpec validate，再启动后台；runner 失败不中断并写 failed list。
- [Risk] MP-DRO group weights 只作用于 beam CE，不覆盖 prototype/KD auxiliary。→ Mitigation：保持最小稳定实现，日志记录权重，后续再决定是否扩展到 prototype loss。
