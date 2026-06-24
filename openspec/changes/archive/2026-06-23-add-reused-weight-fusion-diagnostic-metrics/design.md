## Context

当前 `cnn_hybrid_jepa_visual_prior_sweep` 已经产出一批可复用模型配置和 checkpoint，并且 `kd-sensing-jepa-gps-shortcut-benchmark` 已支持 real-forward、condition cache、P0-P5、Scenario C、Scenario D、CxD 以及 GPS-query advantage slice 的部分逻辑。问题不是缺训练，而是 P0-P5 汇总把图像扰动、GPS 错误和混合扰动压成一维，导致不同融合模型的趋势过于相似。

本 change 的方案是复用现有权重做更正交的离线评估：用同一批模型、同一 split/seed/metric profile，跑少量 CxD 条件和 hard-negative 条件，再输出融合诊断派生指标。所有真实运行产物仍写入 ignored `outputs/`。

## Goals / Non-Goals

**Goals:**
- 复用已有 `config + weights` 跑 real-forward 指标，不重新训练模型。
- 提供一个小型、可复查的默认 condition set，覆盖 clean、GPS 坏、image 坏、双模态坏和 hard negative。
- 输出能解释融合行为的派生指标：`image_rescue`、`gps_rescue`、`fusion_interaction`、paired margin 和 clean drop。
- 让 report 明确区分：P0-P5 是兼容鲁棒性表，CxD/A-slice 是融合机制诊断表。

**Non-Goals:**
- 不新增模型结构、loss、训练 curriculum 或 checkpoint。
- 不重写现有 P0-P5 实现。
- 不恢复退役 KD、HiST、residual 或旧脚本入口。
- 不提交任何 `outputs/`、cache、日志或权重产物。

## Decisions

1. 复用现有 benchmark runner，而不是新建评估 CLI。
   - 方案：在现有 JEPA GPS shortcut benchmark manifest/normalization/aggregation 路径上增加 reused-weight diagnostic profile。
   - 原因：runner 已经处理 model config、weights、real-forward cache、metrics、manifest 和 warnings；新入口会重复这套东西。
   - 备选：写一个独立脚本读取 logits cache。放弃，因为会绕开 real-forward provenance。

2. 默认只跑小切片，不跑完整 C0-C4 x D0-D7。
   - 默认条件：`C0+D0`、`C0+D4`、`C0+D6`、`C3+D0`、`C4+D0`、`C3+D4`、`C4+D6`、`C4+D7`，再加 advantage slice 的 `A0/A1/A2` 或等价 hard-negative 条件。
   - 原因：完整 40 格矩阵成本高，而且当前问题是区分融合机制，不是复刻整张 robustness surface。
   - 备选：只跑完整 CxD。放弃为默认，但保留配置扩展。

3. 指标优先看 paired delta 和派生指标，不再只看条件 DBA 排名。
   - `image_rescue`: GPS 正常时，融合模型相对 image-only 在 image-bad 条件下的收益。
   - `gps_rescue`: image 正常时，融合模型相对 GPS-only 在 GPS-bad 条件下的收益。
   - `fusion_interaction`: 双模态受损 drop 与单模态 drop 之和的差，检测是否出现额外崩塌。
   - 原因：这些指标直接对应融合问题，比 P0-P5 mean 更难被单一模态主导。

4. Claim gate 保守处理。
   - P0-P5 可继续输出和比较，但 report MUST 不把 P0-P5 mean 单独解释为融合机制成立。
   - 只有 reused-weight diagnostic metrics 与 strict comparable baseline 同时存在时，才输出 fusion diagnostic pass/fail。

## Risks / Trade-offs

- [Risk] hard-negative peer 选择可能 fallback，导致 A-slice 难度不足。→ 输出 fallback count、peer pool size 和 not-comparable 状态。
- [Risk] 复用 checkpoint 的 config/split 不完全一致。→ 沿用 comparability keys，任何 mismatch 只允许标记为 not-comparable。
- [Risk] 小切片漏掉完整 CxD 的某些 phase transition。→ 默认小切片用于快速诊断，manifest 保留 full CxD opt-in。
- [Risk] 派生指标命名可能被误读为因果证明。→ report 明确写为 diagnostic evidence，不作为注意力或门控因果证明。

## Migration Plan

1. 增加 reused-weight diagnostic manifest 示例或生成器，指向现有 sweep 的模型配置和权重。
2. 扩展 normalization/aggregation，复用已有 CxD 和 advantage condition 表达。
3. 输出新表：`fusion_diagnostic_metrics.csv`、`paired_margin_by_condition.csv`、`fusion_diagnostic_summary.json`。
4. 更新 report/claim gate 文案，保留旧 P0-P5 表。
5. 用 focused tests 验证 manifest、指标公式、not-comparable/fallback 处理和 OpenSpec。

Rollback：删除该 diagnostic profile 和新增汇总表输出即可；P0-P5 和现有 runner 路径不受影响。
