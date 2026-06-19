## Context

当前 H5/G2/F1 同协议真实 P0-P5 结果显示，`JEPA GPS-query k=4` 的 overall DBA 约 0.8536，`Image ResNet+GPS` 约 0.8502，margin 只有 +0.0035。该结果说明 GPS-query pooler 的 query 数量不是主要瓶颈；当前 `gps_query_attention` 仍然是“GPS 条件特征查询当前 image patch tokens”，并未充分体现 JEPA 论文脉络中的 latent-space prediction、masked target prediction 和 world-model 式 temporal abstraction。

相关论文启发可概括为三点：

1. I-JEPA / V-JEPA 的优势来自在抽象 latent 空间预测缺失或未来表征，而不是像普通 supervised encoder 一样只消费当前观测。
2. 多模态 JEPA / JEPA-MSAC 风格更强调跨模态 token 对齐、时序 block masking 和由 context 预测 target latent。
3. 对 GPS-query 来说，GPS 的合理角色应是“条件化与消歧”，不是在 GPS 错误时直接主导 image latent。

因此本 change 将 GPS-query 从单一 attention pooler 升级为 opt-in 的 Predictive GPS-query++ 架构，并增加 GPS-query advantage 场景，用于检验该归纳偏置是否真的优于 Image ResNet+GPS。

## Goals / Non-Goals

**Goals:**

- 提供一个可配置的 Predictive GPS-query++ 下游架构，使其具备 content-query anchor、GPS-query residual、causal temporal latent prediction 和 reliability-aware gate。
- 扩展评测，使 P0-P5 继续作为主 claim 口径，同时新增能打到 GPS-query 强项的 advantage slice。
- 提供 hard negative difficulty：视觉歧义样本、beam-offset-constrained wrong GPS、GPS async/low-rate + image degradation 组合。
- 保持所有真实 claim 必须 strict comparable：同 split、sample_count、history window、prediction horizon、metric profile、difficulty digest 和 seed。
- 输出 gate/attention/latent consistency diagnostics，解释模型何时使用 current image、temporal predicted latent 或 GPS residual。

**Non-Goals:**

- 不用旧 `C/D0-4` 替代 P0-P5 主结果；C/D 或 CxD 只作为机制诊断和 sanity。
- 不恢复旧 KD、HiST、camera residual、GPS residual 或其它 retired research line。
- 不把 condition id、`P4`、`C4`、`D7` 等字符串直接喂给 gate 作为捷径。
- 不要求改 Stage 1 JEPA checkpoint schema、EMA target encoder 或 pretraining 主流程。
- 不把真实 checkpoint、metrics、PNG、CSV 或 TensorBoard 产物纳入源码变更。

## Decisions

### Decision 1: 保留 P0-P5 主口径，新增 GPS-query advantage slice

P0-P5 继续是 predictive robustness 主 claim 表。新增 GPS-query advantage slice 只回答机制问题：在视觉歧义、wrong GPS、GPS async/low-rate 与 image degradation 组合下，Predictive GPS-query++ 是否比 Image ResNet+GPS 和当前 GPS-query k=4 更稳。

备选方案是直接回退旧 C/D 矩阵作为主表。该方案会更容易制造 JEPA 优势，但会削弱 claim 可信度，也与现有 predictive robustness spec 中“P-suite 是主口径，CxD 是 sanity”的边界冲突。

### Decision 2: GPS-query 变为 residual 条件路径，而非唯一 image latent 路径

Predictive GPS-query++ 使用三条 latent 分支：

- `current_content_latent`: learned content query 或 mean/content anchor，从当前 image patch tokens 得到。
- `gps_residual_latent`: GPS-conditioned query 只生成相对 anchor 的 residual/bias。
- `temporal_predicted_latent`: causal temporal predictor 从历史 image latent 预测当前或未来 latent。

最终 fusion 使用 reliability-aware gate。GPS path 初始权重或 residual alpha 必须受控，避免训练初期 GPS query 完全覆盖 content anchor。

备选方案是继续扩大 `k_queries` 或 attention heads。最近 k=1..5 实验显示该方向只能产生千分位变化，不足以形成稳定优势。

### Decision 3: temporal predictor 从 mean-history 升级为 opt-in causal predictor

当前 `feature_consistency_gate` 中的历史预测主要是历史 latent mean。该方法可诊断但表达力不足。新增 `temporal_predictor` 配置，支持轻量 GRU 或 causal transformer。第一阶段实现可以优先支持 GRU，保持低显存和低复杂度；后续再扩展 transformer。

预测器只可读取当前步之前的 image latent，不得读取未来帧或 target 信息。其输出作为 gate 分支，也可参与 latent consistency / masked target auxiliary loss。

### Decision 4: gate 只消费连续可靠性信号，不消费 condition id

Reliability-aware gate 可以消费：

- `image_valid_mask`
- `image_observability_score`
- `image_current_missing_mask`
- `gps_valid_mask`
- `gps_counterfactual_mask`
- `gps_delay_steps`
- latent consistency scores，例如 `||current - predicted||`、`||gps_residual - current||`

Gate 不得直接消费 `condition`、`predictive_condition_id`、`gps_condition`、`image_condition`、`c_idx`、`d_idx`。Condition metadata 只可进入 diagnostics 和 manifest 归档，用于事后分组。

### Decision 5: hard negative 必须约束 beam offset 和 replay metadata

现有 plausible wrong GPS batch peer replacement 可能替换到 beam offset 很小的样本，难以真正测试 GPS-query 的错导风险。新增 hard negative 选择器必须记录 source sample、scene constraint、GPS distance、beam offset、fallback reason 和 affected sample count。

Visual-ambiguous peer 也应记录 selection basis：同 scene、图像 embedding / metadata proxy 相似、target beam offset 下限、seed 和 fallback。

### Decision 6: 同协议实验分三层

第一层是 synthetic/contract tests：验证 shape、mask、gate metadata、no-future-leak 和 determinism。

第二层是 strict real evaluation：至少包含 `Image ResNet+GPS`、当前 `JEPA GPS-query k=4`、Predictive GPS-query++。同一 H5/G2/F1、scene32-34、future=1、seed=17、P0-P5 与 advantage slice。

第三层是解释性 diagnostics：attention map、gate weight、latent consistency、per-condition margin、target rank CDF 或 top-k movement。解释性图只辅助说明，不单独构成 claim。

## Risks / Trade-offs

- [Risk] 新 advantage slice 被误解为 cherry-pick。  
  → Mitigation: P0-P5 仍为主表；advantage slice 标注为机制诊断，所有报告必须同时列出 clean/P0-P5 和 ResNet baseline。

- [Risk] Gate 通过 condition id 学会捷径。  
  → Mitigation: specs 和 tests 明确禁止 condition id 输入；只允许连续可靠性信号；diagnostics 记录 `condition_id_consumed=false`。

- [Risk] Temporal predictor 增加复杂度但未提升。  
  → Mitigation: 先实现低风险 GRU predictor 和 mean-history baseline 对照；所有配置 opt-in；保留原 GPS-query k=4 baseline。

- [Risk] Hard negative selection 在某些 split 缺少足够 peer。  
  → Mitigation: 支持 deterministic fallback，并在 metrics/warnings 中记录 fallback 比例；fallback 比例过高时 claim 标为 not-comparable 或 warning。

- [Risk] 真实训练成本增加。  
  → Mitigation: 先跑 focused smoke 和单 seed；真实 claim 需要完整 strict run，但源码实现不依赖真实产物。

- [Risk] 输出和诊断产物膨胀。  
  → Mitigation: 真实 CSV/PNG/checkpoint 仍在 ignored `outputs/analysis/...`；源码只提交配置、代码、测试和文档摘要。

## Migration Plan

1. 先增加 spec 和测试，保证旧 `gps_query_attention`、mean pooling、Image ResNet+GPS 配置不变。
2. 新增 opt-in pooler/core/gate/temporal predictor 组件和配置，不改默认入口。
3. 新增 difficulty hard-negative operators 或参数，先通过 synthetic tests 验证 determinism 和 no-future-leak。
4. 新增 H5/G2/F1 Predictive GPS-query++ 配置和 diagnostic manifest。
5. 跑 focused tests，再跑真实训练/eval；真实结果只写入 ignored outputs。
6. 若真实结果不优于 ResNet，保留 capability 为 pending，不升级 claim registry。

Rollback 策略：删除或禁用新 opt-in 配置即可回到现有 GPS-query k=4 和 P0-P5 评测；不需要迁移已有 checkpoint。

## Open Questions

- Hard negative 的 beam offset 下限采用 8、12 还是按 label-space 比例配置，需用小样本统计确认 fallback 比例。
- 第一版 temporal predictor 使用 GRU 还是 causal transformer；建议先 GRU，避免把收益和容量混在一起。
- Advantage slice 是否固定为 8 个条件，还是输出完整 `C3/C4 x D3/D4/D6/D7` 加 P-hard 两组；实现时可先固定 8 个条件。
- 是否为 GPS-query++ 增加 auxiliary latent loss 的默认权重，还是先只做架构/gate 并在第二阶段加入 loss。
