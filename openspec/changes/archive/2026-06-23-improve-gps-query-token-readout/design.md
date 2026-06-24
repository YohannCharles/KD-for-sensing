## Context

当前 `pooler_gps_query_k2_tokens` 的实际结构是：`jepa_context_image` 生成 196 个 patch tokens，`GPSQueryPool` 用 projected GPS feature 生成 2 个 query，对 patch tokens 做 cross-attention，并以 `output_mode: tokens` 保留 `[B,T,2,64]` token 输出；`ModularSequenceModel` 再把 image query tokens 与 GPS frame feature 拼成 token/channel 维，交给一层 `token_aware_transformer`，最后仍对 token/channel 维求均值。

已有 P0-P5 heatmap 显示该路线值得保留，但不值得盲目扩张：Scene31 上 `pooler_gps_query_k2_tokens` 的 overall P0-P5 mean 为 0.638885，明显高于 `pooler_mean` 的 0.532579；但与 `pooler_gps_query_k2_frame` 的 0.638261 基本同档。S31-S34 / S32-S34 汇总中 `k2_tokens` 也优于 `k2_frame` 和 mean，但差距仍需要 seed 和 readout 诊断确认。

约束：

- 当前主线是 Image+GPS JEPA query-pool，不恢复旧 KD、旧 GPS residual 或 retired downstream 路线。
- 默认 `GPSQueryPool` frame 输出和现有 configs 必须保持兼容。
- 训练、评估、attention map、cache、checkpoint 和报告继续写到 ignored `outputs/`。
- 所有项目 Python 验证命令使用 `conda run -n kd_mm_beam ...`。

## Goals / Non-Goals

**Goals:**

- 确认 `k2_tokens` 的收益来自 GPS-conditioned query token 的有效分工，而不是随机 seed、checkpoint selection 或 token-aware core 的偶然正则化。
- 增加最小 token readout 候选，让 K 个 query token 在进入 beam head 前不只被简单 mean 掉。
- 为 query diversity、attention entropy、readout weight、paired DBA/Top-k delta 和失败样本生成 machine-readable 证据。
- 把 `pooler_mean`、`gps_query_k2_frame`、`gps_query_k2_tokens`、新 readout candidate 放入同口径 paired sweep 和 claim gate。

**Non-Goals:**

- 不重写 JEPA Stage 1 预训练、mask sampler、EMA target encoder 或 latent prediction loss。
- 不把 Predictive GPS-query++、geometry prior rerank 或 temporal fallback 合并进本 change。
- 不新增外部依赖，不引入新的大 backbone，不把 `k_queries` 扩到一组大搜索。
- 不用单个 attention overlay 或单 seed 指标宣称架构结论。

## Decisions

### Decision 1: 先诊断 query 分工，再改 readout

第一步不是新增模型，而是让现有 `GPSQueryPool` / `jepa_visual_analysis` 输出 query-level 诊断：

- query attention entropy、peakiness、effective patch count；
- query 间 attention cosine / Jensen-Shannon distance 或等价 diversity；
- query attended latent 的 cosine similarity；
- per-condition 的 query diversity 与 DBA/Top-k delta 相关性；
- regression/gain/far-error case 中 query 热点和 readout 权重。

原因：如果两个 GPS query 学到的是几乎相同的 attention，learned readout 只会多一层噪声；如果二者确实分工，简单 mean 才是需要替换的瓶颈。

备选方案是直接增加 `k_queries` 或 `num_heads`。历史 k=1..5 和当前 k2 token/frame 结果都说明该方向收益很小，先跳过。

### Decision 2: readout 候选保持小而显式 opt-in

新增候选优先级：

1. `gps_query_k2_tokens_weighted_readout`：对 `[B,T,K,D]` query tokens 做轻量 learned query weights，输出 `[B,T,D]`，可记录每个 query 权重。
2. `gps_query_k2_tokens_cls_readout`：添加一个 learned readout token 或等价 query，对 K 个 query token 与 GPS frame token 做一层 attention 后输出 `[B,T,D]`。
3. 保留现有 `gps_query_k2_tokens + token_aware_transformer` 作为 baseline，不删除、不重命名。

实现上优先复用现有 `REPRESENTATION_CORES` 或窄 readout module；如果几行 pooling/readout 能放在 `modular.py` 当前 core 边界内，就不新增单实现 factory。配置必须显式声明 readout 类型，默认仍走现有行为。

### Decision 3: token-aware core 需要记录 readout 语义

当前 `TokenTransformerCore` 最后 `mean(dim=2)`，metadata 只说明 core type，不说明 token readout 是否 learned、uniform mean 或 query-weighted。新契约要求 final config / runtime metadata 至少记录：

- `pooler_output_mode: tokens`
- `k_tokens`
- `token_source`
- `token_readout_type`
- `readout_trainable_params`
- 如果是 weighted/attention readout，记录 query/readout weight summary

这样 heatmap 里同名候选不会只看 `token_aware_transformer` 而丢掉 readout 语义。

### Decision 4: paired claim gate 只比较同口径候选

claim gate 最小比较组：

- `pooler_mean`
- `pooler_gps_query_k2_frame`
- `pooler_gps_query_k2_tokens`
- 新 readout candidate

同一 gate 必须固定 split、scene set、seed、history window、GPS source window、prediction horizon、beam label space、metric profile、distance metric、normalization artifact、difficulty digest、checkpoint selection 和 output root。结论至少报告 Scene31、S31-S34、S32-S34、P0 clean、P1-P5 mean，以及 P3/P4 wrong-GPS / joint degradation 的 delta。

升级条件建议：

- 相对 `gps_query_k2_frame` 的 S31-S34 mean DBA 提升达到最小阈值，例如 +0.005；
- clean/P0 相对 `k2_frame` 不回退超过 0.01；
- P3/P4 或 P0-P5 mean 不出现严重 regression；
- 至少 seed 17/23/42 中多数 seed 同向，或报告 mean/std 后仍正向。

阈值先作为 config 字段，不写死到模型代码。

### Decision 5: attention evidence 是解释，不是训练输入

attention map、query diversity、case panel 和 overlay 只用于诊断与报告。模型 forward 不得读取 target beam、beam power oracle、condition id 或评估结果来决定 readout。P0-P5 condition metadata 只能用于诊断分组、gate 和报告。

## Risks / Trade-offs

- [Risk] learned readout 在小数据上过拟合 clean scene。→ Mitigation：readout 参数量保持极小，必须和 `k2_frame`、`k2_tokens` paired 比较，并报告 Scene31 与 S32-S34 分组。
- [Risk] query diversity 高但与 DBA 无关，attention 解释误导。→ Mitigation：报告 diversity 与 metric delta 的相关性，并保留 regression case，不把 overlay 当因果证据。
- [Risk] token-aware core 当前把 GPS frame token 和 image query token 混在同一 token/channel 维，readout 语义容易混淆。→ Mitigation：metadata 明确 token source 和 readout type，测试锁定 shape 与 summary 字段。
- [Risk] sweep 继续膨胀。→ Mitigation：第一版只加 1-2 个 readout 候选和 seed confirm，不新增大 backbone、不做大规模 `k_queries` 网格。
- [Risk] 旧 config 或 checkpoint 被误解释为新 readout。→ Mitigation：新 readout 必须显式 opt-in；旧 metadata 缺少 readout 字段时按 `uniform_mean` 或 `legacy_token_aware_transformer` 标记。

## Migration Plan

1. 增加 synthetic focused tests，先锁定现有 `k2_tokens` shape、metadata 和默认兼容行为。
2. 增加 readout 候选与 metadata 字段，不改变现有 configs。
3. 扩展 sweep manifest/generator，新增 readout variants 和 claim gate 表。
4. 用已有 outputs 重算或读取 paired summary；真实训练产物仍只写 `outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep/`。
5. 若 readout 候选不稳定，回滚方式是删除新增候选配置和 readout module，保留诊断输出；现有 `pooler_gps_query_k2_tokens` 不受影响。

## Open Questions

- 第一版 learned readout 只读 image query tokens，还是同时读 projected GPS frame token？默认建议同时记录两个版本，但只实现更小的一个。
- claim gate 阈值是否采用 +0.005 DBA，还是沿用现有 GPS-query evidence gate 的阈值字段？
- 是否需要为 query-level contribution 做真正 ablation forward，还是先用 attention/readout/probability proxy？第一版建议 proxy，真实 ablation 只在 proxy 显示有价值时再加。
