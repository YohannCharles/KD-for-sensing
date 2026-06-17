## Context

当前 Image CNN+GPS 是一个强监督、任务校准充分的 baseline，在 uniform C0-C4 x D0-D7 Mean DBA 上已经达到约 0.773。若主目标定义为“全 40 个 cell 上比 CNN+GPS 高 5 个百分点”，目标值会超过 CNN+GPS 的 clean cell 表现，容易把研究问题变成不合理的指标优化。JEPA 的可信优势应来自预测表征：当前图像不可观测或局部被遮挡、GPS 可信但错误、历史视觉上下文可用于恢复当前 beam-relevant latent。

仓库已有基础包括：`jepa_context_image` encoder 的可插拔 pooler/adapter、GPS-query pooler、temporal fallback、modular sequence 模型、shared difficulty pipeline、JEPA GPS shortcut benchmark 与 Scenario D CxD 输出。缺口是这些能力尚未组合成一个“预测鲁棒性”主线，也没有避免 GPS-query 在 mild stale GPS 下过度依赖 GPS 的 hybrid pooler/gate。

## Goals / Non-Goals

**Goals:**

- 新增 Predictive Robustness suite，专门评估“当前 sensing 不完整但历史可预测”的 JEPA 主场景。
- 新增 JEPA predictive hybrid fusion 模型线，使用 mean/content query、GPS residual query、temporal predicted latent 和 feature-consistency gate。
- 主 claim 使用严格可比较的 regional metric：predictive robustness DBA 相对 Image CNN+GPS 提升至少 0.05；同时保留 overall CxD sanity 作为副指标。
- 所有新增扰动复用 shared difficulty pipeline，保持 target、beam power、sample id、split metadata 不变。
- 通过派生配置和包内 benchmark runner 接入，不新增旧式 root script，不复制训练循环。

**Non-Goals:**

- 不要求现有 JEPA GPS-biased 或 JEPA GPS-query-pool 成为最终模型。
- 不把 C/D condition id 直接输入 gate，也不实现 CxD router。
- 不重写 Stage 1 JEPA pretraining、target encoder EMA、mask sampler 或 checkpoint schema。
- 不删除或替换现有 Scenario D CxD、Image CNN+GPS、Image AE+GPS、GPS-only 等对照。
- 不提交真实训练输出、checkpoint、CSV、PNG、cache 或日志。

## Decisions

1. 新增 `predictive_jepa_robustness` suite，而不是改写 `D0-D7`。

   旧 CxD 是通用鲁棒性矩阵，应该继续作为 sanity。新 suite 使用 `P0-P5` 一组预测鲁棒性 condition，例如 clean、current frame missing/history available、semantic occlusion、plausible wrong GPS、joint predictive recovery、novel weather/history available。这样研究 claim 与场景语义一致，也避免把旧 CxD 改成“只服务 JEPA”的指标。

   备选方案是把 `D8/D9/D10` 加进 Scenario D。该方式会混淆 Scenario D 的图像可观测性定义，并让旧 CxD heatmap 的 row/column 语义膨胀。

2. 主模型采用 hybrid pooler，而不是单独 GPS-query pooler。

   当前 GPS-query pooler 能修复一部分 weather/patch selection 问题，但在 stale GPS 下仍可能被 GPS query 带偏。新 pooler 应输出：

   ```text
   z_mean = mean(tokens)
   z_content = learned content-query attention(tokens)
   z_gps = GPS residual query attention(tokens, gps_condition)
   z_img = z_mean + alpha * MLP([z_content - z_mean, z_gps - z_mean])
   ```

   `alpha` 由特征预测或配置初始化，GPS 只能作为 residual bias，不能完全替代视觉全局锚点。pooler 输出仍为 `[B,T,D]`，继续被现有 projector/core/head 消费。

   备选方案是只增加 `k_queries` 或 `num_heads`。这会提高容量，但不能解决 GPS condition 本身可信但错误的问题。

3. temporal predicted latent 作为独立 auxiliary branch 暴露给 downstream fusion。

   `jepa_context_image` 已有 temporal fallback 概念，但 predictive robustness 需要同时保留 `z_current` 和 `z_pred`，而不是只在 encoder 内部替换输出。实现上应让 encoder 在不改变默认输出的前提下记录或返回 `last_current_latent`、`last_temporal_predicted_latent`、history range 和 availability metadata；modular model 或 fusion helper 将其投影后送入 feature-consistency gate。

   备选方案是新增一个 `jepa_predicted` modality。该方式会扩大数据契约和配置校验面，MVP 不需要。

4. feature-consistency gate 不读取 C/D/P condition id。

   Gate 可消费 `z_current`、`z_pred`、hybrid image latent、GPS latent、`image_observability_score`、`gps_delay_steps`、valid masks 等信号，输出 current/predicted/GPS residual 的融合权重和 diagnostics。它 MUST NOT 使用 `c_idx`、`d_idx`、`predictive_condition_id` 作为直接输入。这样它是基于特征一致性的模型架构，不是 CxD router。

   备选方案是直接训练一个 condition-aware router。该方案更易在 P-suite 上过拟合，也会和后续 CxD router change 边界重叠。

5. Predictive Robustness 主指标与 overall CxD sanity 并列。

   Benchmark 输出至少包含 `predictive_dba`、`predictive_top1`、`margin_vs_cnn_dba`、`claim_pass_5pt`、`overall_cxd_dba`、`overall_cxd_delta_vs_cnn`。主 claim 只在 strict comparable、同 split/label/metric/sample_count/difficulty digest 下成立；smoke 或 synthetic runs 只能写 schema/status，不写真实性能 claim。

   备选方案是只报告 worst-case `C4+D7`。该指标太窄，容易变成挑点；regional aggregate 更稳。

6. 训练 profile 必须包含 counterfactual corruption，而不是只 evaluation-only。

   要拉开与 CNN+GPS 的差距，模型需要在训练阶段见到 current image missing、semantic occlusion、plausible wrong GPS、GPS condition dropout 等情况。MVP 提供派生训练配置，真实 claim 需要 train-then-evaluate 或明确 checkpoint provenance；evaluation-only 可用于复用已有 checkpoint 但不能作为最终主 claim。

## Risks / Trade-offs

- [Risk] 新 P-suite 被误读为替代旧 CxD。→ Mitigation：文档和输出同时保留 overall CxD sanity，并在 claim registry 中区分主 claim 与副 sanity。
- [Risk] Hybrid pooler 仍学到 GPS shortcut。→ Mitigation：使用 residual GPS query、GPS condition dropout、plausible wrong GPS training 和 feature-consistency diagnostics。
- [Risk] temporal predicted latent 在历史不足时制造噪声。→ Mitigation：记录 branch availability、history source range、fallback 策略；不足历史时可 raw/skip，并在 metrics 中分组。
- [Risk] P-suite 太窄导致 claim 不够有说服力。→ Mitigation：至少包含 current missing、semantic occlusion、plausible wrong GPS、joint predictive recovery 和 novel weather/history available 多类条件，并保留 strict comparability。
- [Risk] 新 gate 与 CxD router 概念混淆。→ Mitigation：gate 不读取 condition id，只基于 latents/masks/scores；如需 condition-aware routing，留给独立 change。

## Migration Plan

1. 新增或扩展 JEPA downstream pooler：注册 `hybrid_residual_query` 或等价名称，保持 mean/GPS-query 默认行为不变。
2. 扩展 `jepa_context_image` temporal auxiliary branch，暴露 current/predicted latent metadata，不改变默认 checkpoint 加载和 forward 输出。
3. 新增 feature-consistency fusion helper 或 modular representation core，优先作为可组合组件接入 `modular_sequence`。
4. 新增 predictive robustness difficulty presets/operators 和 tests，保证 no-label-shift、determinism、history no-future-leak。
5. 新增 JEPA predictive hybrid fusion 派生配置和 predictive robustness benchmark manifest。
6. 扩展 benchmark runner 的 suite 解析、aggregation、CSV/JSON/manifest 输出和 smoke tests。
7. 更新实验矩阵、模型目录、实验协议和 claim registry；真实产物只写入 ignored outputs。

Rollback 策略：移除或停用派生配置中的 hybrid pooler/gate 与 predictive suite，旧 JEPA GPS-biased、GPS-query-pool 和 Scenario D CxD workflow 继续按原路径运行。

## Open Questions

- `P` conditions 的首版数量是否固定为 P0-P5，还是先实现 P0-P4 MVP 后再扩展 novel weather。
- feature-consistency gate 是否放在 `jepa_downstream.py` 的 adapter 后，还是作为 `REPRESENTATION_CORES` 的新 core；实现时需以最少破坏 modular 边界为准。
- 真实 claim 使用的训练预算、checkpoint provenance 和 seed 数量需要在首次 real run 前确定。
