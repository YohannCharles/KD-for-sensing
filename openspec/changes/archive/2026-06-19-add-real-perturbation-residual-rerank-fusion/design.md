## Context

上一轮 `add-geometry-prior-beam-fusion` 的结果显示，geometry prior 路线能守住 clean/P0，但没有显著超过 `Image ResNet+GPS`：clean/test DBA 分别约为 baseline 0.8857、prior-only 0.8855、logit fusion 0.8833。专用 claim gate 已保持 `pending`，因为 P-suite/advantage rows 由 benchmark runner 的 deterministic degradation model 生成，并非真实逐条件 forward。

这暴露出两个问题：第一，没有真实 perturbation forward，就无法判断 geometry prior 是否真的在 wrong GPS、missing image 或 joint degradation 下有帮助；第二，普通 additive logit fusion 会把 geometry prior 当作全局信号加入，容易在 clean 条件上被强 baseline 抵消，或者在错误 GPS 下变成噪声。下一步应把强 baseline 作为 anchor，用 geometry prior 做“候选集内的安全重排”，并用真实扰动评估证明是否值得升级 claim。

## Goals / Non-Goals

**Goals:**

- 支持 real-forward benchmark：对 P0-P5、GPS advantage、CxD 或指定 difficulty condition 真正应用 batch transform、执行 checkpoint forward、保存 logits/labels/diagnostics。
- 构建 safe residual beam rerank fusion：以 `Image ResNet+GPS` 或同构 image+GPS branch 的 logits 为 anchor，只在 candidate beam set 中做 bounded residual/rerank。
- 保护 clean：clean/high-observability 条件下默认 fallback anchor，reranker 必须通过 no-regret gate 才能改变 top prediction。
- 让 prior 用得可解释：输出 candidate set 来源、prior rank、anchor rank、rerank residual、fallback reason、branch entropy/agreement、real perturbation metadata。
- 将 primary claim gate 与 evidence scope 绑定：只有真实 per-condition forward metrics 可以升级 claim；synthetic/delegated degradation 只能作为机制诊断。

**Non-Goals:**

- 不恢复旧 DeepSense6G GPS residual fusion、Top8 selector、camera residual、HiST、KD 或历史兼容 wrapper。
- 不新增根脚本或复制训练循环；模型仍优先作为 `modular_sequence` component baseline。
- 不承诺一次实现完整 BEV-Fusion 论文复现；如要做 BEV 空间主线，应另起 workflow/paper reproduction change。
- 不使用 target_test label、beam power oracle、future frame 或 condition id 作为模型输入。

## Decisions

### Decision 1: 真实扰动 forward 优先于更多合成 degradation 估算

Benchmark runner 增加 real-forward 模式：每个 model、condition、seed 和 split 都构造 dataloader，调用统一 difficulty pipeline 变换 batch，然后走标准 `forward_model` / evaluator 计算 logits 和 metrics。输出 per-condition logits cache、metrics CSV、warnings 和 replay metadata。

备选方案是继续使用 deterministic degradation model。它快，但已经导致 claim gate 不可升级，也无法验证分支诊断和 fallback 是否真实生效。

### Decision 2: reranker 只在候选 beam set 内做安全重排

模型先生成 anchor logits，再构造 candidate set：

1. anchor top-k；
2. geometry prior top-k；
3. anchor top-1 附近的 beam 邻域；
4. 可选 teacher/EMA top-k。

reranker 只对这些候选 beam 产生 residual score，并通过 mask 写回全 64 beam logits。非候选 beam 保持 anchor logits 或低权重 residual，避免大范围扰动 clean 分布。

备选方案是再做全量 additive logit fusion。上一轮结果显示全量 additive fusion没有明显收益，且难以解释哪些 beam 被 geometry prior 帮助。

### Decision 3: no-regret gate 约束 clean 和高不确定场景

Reranker 需要输出 `fallback_to_anchor` 和原因，例如 low confidence、prior-image disagreement、GPS invalid、image high observability + anchor confidence high。训练时可加入 anchor consistency / no-regret loss；推理时 residual magnitude 受 `max_residual_scale` 限制。

备选方案是让 gate 学习任意权重。这样可能提升少数 hard cases，但 clean regression 风险高，且容易偷用 condition id 或过拟合 difficulty pattern。

### Decision 4: 训练目标使用 pairwise/ranking + hard beam supervision

主 loss 仍保留 hard-label CE 或 focal CE。新增可选 rerank objective：

- candidate cross-entropy：只在 candidate set 内预测 target；
- pairwise margin：target beam 或 DBA-near beam 分数高于 anchor top-1；
- no-regret consistency：anchor 正确或高 DBA 时，rerank logits 不应明显破坏 anchor。

DBA-aware smoothing 可作为辅助，但不替代 hard target metrics。

### Decision 5: diagnostics 是模型接口的一部分

模型 forward 输出必须可被 `adapt_model_output` 消费，同时附加 diagnostics：

- `anchor_logits`、`rerank_logits`、`geometry_prior_logits`；
- candidate beam ids、candidate source mask、selected source；
- anchor/proir/rerank top-k 和 target rank；
- residual score statistics、fallback reason、gate confidence；
- condition id consumed flag，必须为 false。

这些字段可选传入 diagnostics aggregator；普通 baseline 不需要输出或消费这些字段。

### Decision 6: claim gate 分层

Claim gate 输出至少分三层：

- `clean_gate`: 真实 clean/P0 forward 是否在阈值内；
- `real_forward_p_suite`: P0-P5 是否来自真实扰动 forward；
- `primary_claim`: 只有 clean gate 通过且 real-forward P-suite/advantage 达到 margin 时才 pass。

如果只有 delegated clean-only 或 synthetic degradation，则 status 必须是 `pending`，并写出原因。

## Risks / Trade-offs

- [Risk] real-forward benchmark 很慢。  
  -> Mitigation: 支持 condition subset、sample_count cap、logits cache reuse、resume 和 per-model/condition shard。

- [Risk] reranker 过度依赖 GPS prior，clean 或 wrong GPS 下退化。  
  -> Mitigation: no-regret gate、bounded residual、GPS reliability fallback、clean regression gate 和 condition id isolation tests。

- [Risk] candidate set 漏掉 target beam，导致 reranker上限受限。  
  -> Mitigation: 诊断 candidate recall@K、prior/anchor target rank；训练可逐步调整 K 和邻域宽度。

- [Risk] real-forward evaluation 误用 target labels 或 future information。  
  -> Mitigation: difficulty pipeline 保持 target label 不变，模型输入过滤 target/test-only fields，新增 leakage tests。

- [Risk] “residual” 名称被误解为恢复旧 GPS residual research line。  
  -> Mitigation: spec 明确这是 component-level bounded rerank residual，不恢复 retired workflow、旧 CLI、旧 configs 或 Top8 selector。

## Migration Plan

1. 扩展 benchmark runner 的 real-forward evaluation path，先用现有 `Image ResNet+GPS` 和 geometry-prior checkpoint 复现 clean metrics，再启用一个 P0/P1 小子集 smoke。
2. 增加 logits/diagnostics cache schema 和 aggregator，确保 delegated/synthetic/real-forward evidence scope 被 machine-readable 标记。
3. 实现 safe residual reranker component 和 synthetic forward tests，覆盖 candidate set、bounded residual、fallback、condition id isolation。
4. 增加训练配置：anchor-safe clean smoke、rerank-only、geometry-prior candidate union、no-regret loss ablation。
5. 跑 strict real-forward matrix，先判断 clean/P0，再跑 P-suite 和 GPS advantage。
6. 只有 real-forward claim gate 通过时，才更新 result claim registry 或 mainline docs；否则记录 pending/failed。

Rollback：禁用 reranker config 即回到现有 `Image ResNet+GPS`、JEPA GPS-query 和 geometry-prior configs；real-forward benchmark 只写 ignored outputs，不改变默认训练入口。

## Open Questions

- anchor logits 是复用同一模型内部 image+GPS branch，还是先加载外部 frozen `Image ResNet+GPS` checkpoint 作为 teacher/anchor？
- candidate set 默认 K 取 8、12 还是 16，beam 邻域宽度取 ±1 还是 ±2？
- no-regret gate 在训练期使用 hard oracle 判断 anchor 是否正确，还是只用 DBA-near soft target？
- real-forward strict matrix 的 first pass 是否只跑 P0-P5，还是同时跑 GPS advantage 8 个 combined conditions？
