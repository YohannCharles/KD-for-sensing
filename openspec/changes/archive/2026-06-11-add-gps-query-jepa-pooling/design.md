## Context

当前 GPS-conditioned JEPA Stage 1 已能用 GPS-Rel-Polar 对 latent prediction 做条件化，并通过 `context_encoder` checkpoint 提供下游 supervised image encoder。Stage 2 的 `jepa_context_image` 目前将 `VisualPatchTokenEncoder` 输出的 patch tokens `[B,T,N,D]` 直接 `mean(dim=2)` 为 `[B,T,D]`，再与 GPS encoder 输出交给 `modular_sequence` 的 fusion core。`fair_gps_biased` 是当前 Image+GPS+JEPA 主线结果，但 mean pooling 会把局部目标位置、遮挡和道路几何线索平均掉。

本设计把 GPS-query Attention Pooling 放在 Stage 2 的 JEPA context image encoder 内部：JEPA patch tokens 仍来自 frozen 或 trainable `context_encoder`，GPS 条件来自当前下游 `gps_mlp` + projector 后的 `[B,T,d_model]` 表征。这样不改变 Stage 1 预训练、不改变 checkpoint schema，也不把 pooling 逻辑塞进 fusion core。

## Goals / Non-Goals

**Goals:**

- 在 `jepa_context_image` 中新增显式 opt-in 的 `pooling: gps_query_attention`，替代 mean pooling 得到 `[B,T,D]` image feature。
- 用下游 GPS/motion 表征生成 `K` 个 query，让 query attend 到每帧 JEPA patch tokens `[B,T,N,D]`。
- 保持 `pooling: mean` 默认行为、现有配置、现有 checkpoint 加载和现有 downstream ablation 可继续运行。
- 基于 `fair_gps_biased` 新增 GPS-query pooling 配置，复用 GPS-biased JEPA checkpoint、BeamBench-fair/2604-style 数据口径和 supervised beam recipe。
- 写出 metadata 与 attention diagnostics，便于区分 mean pooling、GPS-query pooling、query 数量和条件来源。

**Non-Goals:**

- 不重训或修改 JEPA Stage 1 预训练目标、mask sampler、EMA target encoder 或 checkpoint 格式。
- 不引入 TokenLearner 外部依赖；第一版使用 PyTorch `nn.MultiheadAttention` 实现最小 GPS-query pooling。
- 不恢复 GPS residual、Camera residual、Top8 selector、HiST/Hist、KD/distillation 或旧 fusion 入口。
- 不改变 dataset GPS feature mode；默认仍使用 `relative_polar` 与既有 GPS normalizer。

## Decisions

### Decision 1: Query 条件源使用 projected GPS feature

`GPSQueryPool` 的默认条件输入使用 `gps_mlp` 编码并经 projector 对齐后的 `[B,T,d_model]` GPS feature，而不是 raw `[B,T,3]`。模块内部通过 `gps_to_q` 将 `condition_dim` 映射到 `k_queries * latent_dim`，因此即使 `d_model != latent_dim` 也能工作。

替代方案是在 `jepa_context_image` 内部直接读取 raw GPS 并自带 GPS MLP。该方案实现更局部，但会复制 GPS encoder 逻辑、绕开现有 normalization/projector 契约，也难以扩展 motion feature。使用 projected GPS feature 更符合“Stage 2 用当前几何表征引导视觉 token pooling”的边界。

### Decision 2: Pooling 放在 `jepa_context_image` 内部

`JepaContextImageEncoder.forward()` 在 `pooling: mean` 时保持 `tokens.mean(dim=2)`；在 `pooling: gps_query_attention` 时调用 `GPSQueryPool(tokens, gps_condition_features)`，输出仍是 `[B,T,D]`，使下游 projector 与 fusion core 无需改变。

替代方案是让 representation core 消费 `[B,T,K,D]` 或全部 `[B,T,N,D]` tokens。该方案表达力更强，但会改变模块化模型中 encoder/projector/core 的既有 `[B,T,D]` 契约，影响范围更大。第一版先保持帧级 image feature 契约；未来如果 K-token fusion 有实证收益，再单独走 OpenSpec。

### Decision 3: `ModularSequenceModel` 增加 dependency-aware encoder context

`JepaContextImageEncoder` 在 GPS-query pooling 下声明 `required_context_modalities = ("gps",)` 和 `context_feature_source = "projected"`。`ModularSequenceModel` forward 时先编码并投影没有未满足依赖的模态，随后将已投影 GPS feature 通过关键字参数传给需要条件输入的 image encoder，例如 `gps_condition_features=projected["gps"]`。

为保持通用性，模块化模型只支持由 encoder 显式声明的条件依赖；普通 encoder 继续单参数调用。若依赖模态未启用、batch/time 不匹配，或出现循环依赖，系统必须抛出清晰错误。

### Decision 4: `GPSQueryPool` 第一版做 K-query 后平均

第一版 `GPSQueryPool` 使用：

- `gps_to_q`: `LayerNorm/Linear/GELU/Linear` 将 GPS 条件映射为 `[B*T,K,D]` query。
- `nn.MultiheadAttention(d_model=D, num_heads=n_heads, batch_first=True)` 对 `[B*T,N,D]` patch tokens 做 cross-attention。
- `LayerNorm` 后默认对 `K` 个 query token 求均值，返回 `[B,T,D]`。
- 可选返回平均到 head 后的 attention map `[B,T,K,N]`，用于 diagnostics 或可视化。

替代方案是直接输出 `[B,T,K,D]` 给 token transformer。该方案需要修改 projector/core/head 对 K-token image representation 的理解，超出本次“基于 fair_gps_biased 最小改造”的范围。

### Decision 5: 配置以 `fair_gps_biased` 派生，不替换原配置

新增配置从 `image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml` 或同口径 2604 配置继承，只覆盖 image encoder pooling、GPS-query pool 参数、run name 和 metadata 标识。原 `fair_gps_biased` mean-pooling 配置保留为 baseline，便于成对对比。

默认参数建议：

- `pooling: gps_query_attention`
- `gps_query_pool.k_queries: 4`
- `gps_query_pool.num_heads: 4`
- `gps_query_pool.condition_source: projected_gps`
- `gps_query_pool.return_attention: true` 仅用于诊断路径；训练主输出仍为 `[B,T,D]`

## Risks / Trade-offs

- [Risk] GPS feature 可能过强，attention pooling 学成 GPS-only shortcut。→ Mitigation：保留 mean-pooling baseline，新增配置必须报告 image encoder pooling、attention entropy/peakiness，并与 GPS-only 或 existing fair result 对照。
- [Risk] 双阶段依赖编码让 `ModularSequenceModel.forward()` 复杂度上升。→ Mitigation：只支持 encoder 显式声明依赖；普通 encoder 路径保持原逻辑；focused tests 覆盖未声明与缺失依赖。
- [Risk] `k_queries` 后平均仍可能损失多目标/多遮挡局部结构。→ Mitigation：第一版记录 attention map；若 attention 呈现多区域稳定模式，再另开 K-token fusion change。
- [Risk] 使用 projected GPS feature 要求 GPS 模态启用并与 image 时间维对齐。→ Mitigation：配置加载与 forward 必须在 GPS 未启用或 batch/time 不一致时报错。
- [Risk] attention map 写入训练输出可能增加内存。→ Mitigation：训练 forward 默认只返回轻量 diagnostics；大尺寸 attention map 仅在显式 debug/diagnostic 配置下返回或 detach 后下采样记录。

## Migration Plan

1. 新增 `GPSQueryPool`、扩展 `JepaContextImageEncoder` pooling 配置与 metadata。
2. 扩展 `ModularSequenceModel` 的 encoder context 调用，保持普通 encoder 单输入兼容。
3. 新增 `fair_gps_biased` 派生配置和 README 说明，原配置不改名、不删除。
4. 新增 focused tests：pooling shape、缺失 GPS 条件、mean 默认兼容、modular image+GPS forward、配置加载和 runtime metadata。
5. 运行 OpenSpec 校验与相关测试：
   - `openspec validate add-gps-query-jepa-pooling --strict`
   - `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py -q`
   - 必要时补充 `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py -q`

Rollback 方式是删除新增 GPS-query pooling 配置、模块、tests 和 OpenSpec change；已有 mean-pooling JEPA downstream 配置与 checkpoint 不需要迁移。

## Open Questions

- 第一轮实验是否只在 BeamBench-fair lowmem 口径上跑，还是同时补 2604 S32/S33/S34 macro 口径配置。
- attention diagnostics 是否需要写入 viewer manifest，还是先保留为模型输出/单测诊断字段。
- 如果 GPS-query pooling 提升明显，下一步是否扩展为 `[B,T,K,D]` K-token representation 并接入 `next_beam_query_transformer`。
