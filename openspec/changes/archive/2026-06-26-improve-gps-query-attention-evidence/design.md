## Context

当前仓库已经有三层相关能力：

- `GPSQueryPool` 能在 opt-in attention diagnostics 下记录 `[B,T,K,N]` query-to-patch attention map。
- `kd-sensing-jepa-visual-analysis` 能导出 patch-grid heatmap、query/time 分面图、image-space overlay、attention summary、case panel 和 report。
- `gps-query-effectiveness-visualization` 已把 paired ablation、case selection、claim gate 和 attention caveat 纳入证据包边界。

问题不在于没有图，而在于图的解释口径仍然太容易被过度解读。当前 overlay 展示的是 GPS 条件 query 对视觉 token 的读取权重；它经过 head 平均、time/query 聚合、上采样和 per-sample minmax 归一化后，不能直接说明“哪些像素导致 beam 预测”。本地已有 attention summary 也显示 query diversity 极低、effective patch count 偏大，说明许多图更像宽泛读取区域，而不是明确局部归因。

因此本 change 将 attention overlay 降级为可审计的 token-read 诊断，并增加最小 faithfulness 检查：如果高 attention patch 的遮挡不比低 attention 或随机 patch 更影响目标指标，报告必须明确写为 exploratory 或 insufficient。

## Goals / Non-Goals

**Goals:**

- 明确 GPS-query attention 图的语义边界：默认称为 `token_read_map`，不作为 causality 或 attribution 证据。
- 在现有 JEPA visual analysis/evidence package 中增加 deterministic patch occlusion faithfulness 诊断。
- 将 faithfulness 结果纳入 claim gate，使 attention 只有在通过检查时才支持解释性结论。
- 在 manifest、CSV 和 report 中记录 attention provenance：shape、token grid、head/time/query 聚合方式、归一化方式、底图来源和遮挡策略。
- 为 GPS-query 类 pooler 增加轻量 metadata，记录是否跨 head 平均，并允许 opt-in per-head attention 诊断。
- 保持现有训练、评估、checkpoint、默认配置和输出目录边界不变。

**Non-Goals:**

- 不把 attention 图变成主证据；主证据仍是 paired ablation、P0-P5/Scene 切片、扰动和 case coverage。
- 不新增默认模型架构、不改变 `gps_query_attention`、`predictive_gps_query` 或 mean pooling 的训练语义。
- 不强制引入 Grad-CAM、LRP、Transformer relevance propagation 等新依赖；可作为 optional fallback 或未来扩展。
- 不恢复 viewer manifest、Gradio viewer、旧 visualization alias 或退役研究路线。
- 不要求读取真实 `dataset/` 来完成单元测试；测试使用 synthetic/mock tensors、logits 和 metadata。

## Decisions

### Decision 1: 将 attention overlay 定义为 token-read map

实现中保留现有 overlay 产物，但在 metadata/report 中统一标记：

- `map_semantics=token_read_map`
- `causal_claim=false`
- `attention_source=gps_query_pooler` 或具体 pooler type
- `aggregation_method`，例如 `mean_head_time_query`、`mean_time_query`、`query_0_time_0`、`head_2_query_1_time_0`

理由：这是最小改动，能立即减少误读。直接重命名文件或删除现有图会破坏已有报告习惯，收益不如在 manifest/report 中明确语义。

备选方案是完全移除 attention overlay，只保留 quantitative ablation。该方案最安全，但会失去调试 token reshape、query collapse 和局部读取模式的实用价值。

### Decision 2: Faithfulness 先做 patch-level deterministic occlusion

第一版 faithfulness 不做复杂解释算法，只比较三组 patch：

- `top_attention`: 按聚合后的 token-read score 选 Top-p 或 Top-k patch。
- `low_attention`: 在非 top 区域中选低 attention patch，patch 数与 top 组一致。
- `random`: 用固定 seed 选择相同 patch 数，支持多次 repeat 后取均值。

遮挡策略优先复用现有 image tensor 和 transform 尺寸，默认使用 `zero` 或 `dataset_mean` 替换；输出比较目标包括 target logit、target margin、Top-k hit、DBA contribution 或可用等价指标。

理由：遮挡检查能直接回答“热区被移除后模型是否更受影响”。它比只看 entropy/query diversity 更接近解释可信度，也不需要新依赖。

备选方案是实现 Grad-CAM 或 Transformer relevance propagation。它们更接近 attribution，但需要额外 hook/gradient 路径和可选依赖处理；第一版先作为 optional comparison，不作为必需项。

### Decision 3: Claim gate 不因 attention 好看而升级

Claim gate 的输入顺序：

1. strict comparability 和 paired delta。
2. clean/P0 regression 与 P1-P5 robustness。
3. deterministic case coverage，必须包含 gain 和 failure/regression。
4. attention 可用性和 faithfulness。
5. query diversity、effective patch count、center spread 等解释性统计。

只有 paired delta 已支持、且 attention faithfulness 通过时，报告才可写“attention token-read pattern supports the interpretation”。如果 paired delta 不支持，即使 overlay 看起来合理，也只能写 exploratory diagnostic。

理由：这延续现有证据包设计，避免把辅助诊断反客为主。

### Decision 4: Pooler 只补诊断 metadata，不改训练 forward

`GPSQueryPool` 和相关 GPS-query 类 pooler 保持默认 `average_attn_weights=True` 行为。新增配置只允许在诊断路径 opt-in：

- 记录 `attention_head_aggregation=averaged` 或 `per_head`。
- 记录 `attention_return_shape` 和 `attention_diagnostics_shape`。
- 可选 `return_attention_heads` 或等价诊断配置，使 `MultiheadAttention` 在分析时返回 per-head attention。

理由：训练 checkpoint 和现有配置必须稳定。per-head attention 只对解释分析有用，不应改变主模型输出或 loss。

备选方案是默认改成 per-head attention。该方案会增加显存和输出体积，也可能改变外部测试对 shape 的假设，不值得。

### Decision 5: 输出继续复用现有 visual analysis/evidence 结构

新增产物沿用现有目录：

- `tables/attention_faithfulness.csv`
- `figures/attention_faithfulness/`
- `cases/*.json` 中增加 `attention_faithfulness` 字段
- `analysis_manifest.json` 或 `evidence_manifest.json` 增加 `attention_provenance`、`faithfulness_summary` 和 warnings

理由：少建入口，少建文件结构。现有 CLI 已负责只读输入、ignored output 和 report 汇总。

备选方案是新增独立 CLI。该方案会重复模型加载、dataset、metrics 和 manifest 逻辑，后续维护成本高。

## Risks / Trade-offs

- [Risk] 遮挡 patch 可能产生分布外输入，导致 faithfulness 结果保守或噪声大。  
  Mitigation: manifest 记录遮挡策略、patch 比例和 repeat seed；claim gate 只把它作为解释性诊断，不作为独立性能证据。

- [Risk] 使用模型输入 tensor 作为底图时，反归一化或 resize 可能降低人类可读性。  
  Mitigation: overlay metadata 必须记录 `overlay_image_source`，报告将该类图标记为调试图而非论文级原图证据。

- [Risk] per-head attention 输出增加内存和 cache 体积。  
  Mitigation: 仅在 analysis config 显式开启时生成，并受 `max_attention_cases` 限制。

- [Risk] 现有本地报告或脚本依赖旧字段名。  
  Mitigation: 旧字段保留，新字段追加；文件名尽量保持兼容，只在 manifest/report 中新增语义字段。

- [Risk] Attention faithfulness 通过但 paired delta 不支持，用户仍可能过度声称。  
  Mitigation: claim gate 强制 paired delta 优先；report 必须把这类结果写入 `interpretive` 或 `caveat`，不得进入 `reportable`。

## Migration Plan

1. 增加 delta specs 和 focused tests，先锁定 token-read 语义、faithfulness 表结构、claim gate 降级和 pooler metadata。
2. 在 visual analysis/evidence helper 中实现 patch selection、occlusion forward、指标比较和 CSV/manifest 写出。
3. 扩展 report builder，新增 attention provenance 与 faithfulness 小节，并保留旧 attention summary。
4. 可选扩展 GPS-query pooler 诊断 metadata；默认仍返回平均后的 attention map。
5. 用 synthetic/mock tests 验证，不读取真实 `dataset/`，不提交 `outputs/`。
6. 回滚时删除新增 opt-in config 字段和 faithfulness 输出 helper；旧 overlay、attention summary 和 paired evidence 不受影响。

## Open Questions

- 第一版遮挡比例默认使用 Top 10%、Top 20%，还是按固定 patch 数，例如 16/196？建议先配置化，默认 10%。
- `dataset_mean` 遮挡是否已有可靠反归一化/归一化 profile 可复用；如果没有，先用 zero/masked token 并记录策略。
- per-head attention 是否放在 pooler config，还是只由 visual analysis 的临时诊断上下文打开；建议先放在 analysis config，避免训练配置面膨胀。
