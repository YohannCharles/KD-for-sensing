## Context

当前 Scene31-34 只有 quick seed1 结果，输出位于 `outputs/scenes31_34_subset_reliability_lmdb`。已完成的 quick run 表明 `proto_randomdrop_subset_es40` 是 pooled winner，但结论仍依赖单 seed，且缺少按 missing_count=0/1/2/3 聚合的 Top1、Within@3 和 MAE 退化曲线。

仓库已有 Scene31/Scene31-34 local/manual runner、apples-to-apples fresh eval helper、maskfix eval 和 summary 脚本。现有规则要求训练与评估继续通过 `conda run -n kd_mm_beam kd-sensing-train` 和保留 fresh eval helper 执行，脚本只做 manifest、队列、路径解析、跳过/覆盖策略和结果汇总，不复制 DataLoader、模型加载或指标计算逻辑。

## Goals / Non-Goals

**Goals:**

- 将 Scene31-34 pooled training/eval 作为缺失模态论文主实验 workflow。
- 补齐 prototype 主 baseline seed1/2/3，其中 old-root seed1 可只读复用，新 root 只承载新训练和新汇总。
- 让 fresh eval 写出 per-sample prediction、pattern metrics、apples-to-apples metrics 和 checkpoint manifest，字段足以生成 missing_count 曲线。
- 生成主 summary、per-scene summary、missing_count degradation curve、论文主表/消融表和最终 conclusion。
- 明确冻结主方法为 `proto_randomdrop_subset_es40`，同时如实报告 seed1/2/3 后的实际 winner。

**Non-Goals:**

- 不继续 reliability fusion seed2/3、PatternFiLM、JTT、MVFR、MPDRO、beamsoft、condBTAPA 或 weakKD。
- 不把 AMR/AMBER-lite 多场景结果作为阻塞项；若配置不可用，只记录可选 next step。
- 不重训 old-root 中可读的 seed1。
- 不提交真实训练输出、checkpoint、logs、figures 或 paper table runtime artifacts。
- 不新增绕过 `src/kd_sensing` 包结构的训练业务逻辑。

## Decisions

### 1. Scene31-34 主实验保持 local/manual script surface

新增 `scripts/run_scenes31_34_main.sh` 作为薄 runner，继续调用 `kd-sensing-train` 和既有 fresh eval helper。这样可以复用当前训练、checkpoint、maskfix、metrics 和 dataset contract，同时符合 inventory 中 Scene31 local/manual runner 的边界。

替代方案是新增 package CLI；本轮只是本地论文主实验队列，不需要长期稳定 API。若后续需要公开复现，可再通过单独 change 收敛为包内 manifest runner。

### 2. old-root seed1 只读复用，新 root 汇总跨 root

summary 和 eval runner 都支持 `--root` 与 `--old-root`。natural、uniform、subset 的 seed1 默认从 old-root 读取；Bernoulli seed1 若 old-root 不存在，则在 new root 补跑。这样避免覆盖 quick seed1 产物，也能让主 summary 输出 seed1/2/3 的统一表。

### 3. missing_count 曲线从 pattern/per-sample fresh eval 派生

fresh eval 必须输出 `predictions_by_pattern.csv` 与 `pattern_metrics.csv`，每行记录 scene、pattern、missing_count、missing_ratio、available_modalities 和 missing_modalities。summary 优先从这些结构化文件生成 missing_count curve；若旧 seed1 缺少新字段，summary 可从 pattern 名称和固定四模态顺序推导，并写 warning。

### 4. 主表只推广 prototype + random subset exposure

主表固定包含 Natural、Uniform pattern exposure、Bernoulli randomdrop 和 Random subset exposure。Reliability fusion 只能作为 seed1 auxiliary curve 或 quick check 背景，不进入主 winner 叙述；PatternFiLM 和其它搜索线不进入本轮输出。

### 5. 论文产物生成脚本只写 ignored output root

summary、figures、paper tables 和 final conclusion 默认写入 `outputs/scenes31_34_main_lmdb/` 或 `outputs/paper_tables/scenes31_34_main/`。源码只新增脚本和文档契约，不纳入生成 CSV/PNG/PDF/MD/TXT。

## Risks / Trade-offs

- old-root seed1 缺少新 per-sample 字段 -> summary 使用可推导字段补齐 missing_count，并在 warnings/conclusion 中记录 provenance；需要严谨图表时可对 seed1 运行 `eval_proto_all` 重评。
- 长时间训练无法在普通验证中完成 -> runner 支持 dry-run/skip/complete 检查，源码验证只跑 help、static/smoke 和 summary fixture；真实训练由用户按推荐命令执行。
- 不同 run 的 fresh eval 字段命名历史不一致 -> summary 实现保留窄 adapter，统一到 method、seed、scene、missing bucket 和核心指标列。
- AMR/AMBER-lite 多场景 maskfix 口径不成熟 -> 本轮不默认运行，只在 paper notes/final conclusion 标记为可选 next step。
- classifier baseline 需要和 proto baseline 共用训练/评估路径 -> 只在生成配置中关闭 prototype alignment 并使用 `loss.type=cross_entropy`，不新增模型业务逻辑或旧入口。
- compute profile 优先读取已有 checkpoint、train_log 和 eval manifest；缺少训练耗时、GPU memory 或 latency 记录时输出 NaN 与 warning，不触发重训或正式 eval。

## Migration Plan

1. 新增 OpenSpec spec 和任务清单。
2. 读取现有 Scene31-34 quick runner、fresh eval helper、summary helper 和 manifest/config 模式。
3. 新增或扩展 runner、summary、plot、paper table 和 conclusion 脚本。
4. 为 summary/plot/table/conclusion 增加小型 fixture 或无数据 smoke 测试，避免依赖真实 `dataset/` 和训练输出。
5. 运行 `openspec validate promote-scenes31-34-main-missing-count --strict` 与相关 focused tests。

Rollback 策略：删除本 change 新增脚本和文档增量即可恢复到 Scene31-34 quick seed1 workflow；不会修改或覆盖 old-root 训练产物。
