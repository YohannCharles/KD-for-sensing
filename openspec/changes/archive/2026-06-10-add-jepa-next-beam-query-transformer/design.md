## Context

当前 JEPA 线由两段组成：上游 `gps_conditioned_jepa` 预训练学习 image patch latent，下游 supervised fusion 通过 `jepa_context_image` 从 checkpoint 抽取 `context_encoder`，将图像序列压成 `[B, T, D]` 帧级特征，再与 GPS MLP 特征一起进入模块化 `representation_core` 和 beam head。

下游 JEPA fusion 当前主要沿用 `early_concat_gru`：image/GPS 分别编码和投影后，在每个时间步拼接为 `[B, T, K*D]`，再由 GRU 输出 `[B, T, hidden]`。这个路径简单可靠，但对“下一时刻单步 beam prediction”有两个局限：一是模型语义仍像逐时刻分类器，而不是显式查询未来一步；二是模态与时间关系被压进拼接向量，缺少可解释的 token 级交互结构。

仓库已有 `modular_sequence`、`snapshot_frame` 和 `token_transformer` core，说明 encoder/projector/core/head 边界已经适合扩展。这个 change 应沿用该边界，只新增下游融合 core 和配置，不改变 JEPA 预训练、数据集字段、beam label space 或训练主循环。

## Goals / Non-Goals

**Goals:**

- 保留 JEPA context image encoder 作为图像分支，不重新设计 JEPA 预训练模型。
- 新增 `next_beam_query_transformer` representation core，用 learned `[NEXT_BEAM]` query 从历史 image/GPS token 中读取下一时刻 beam 表征。
- 显式使用 time embedding 和 modality embedding，让 Transformer 能区分时间顺序与模态来源。
- 输出单步 `[B, 1, D_out]` 表征，并继续复用现有 `beam_head` 生成 `[B, 1, 64]` logits。
- 提供四组可复现实验配置：当前 GRU、snapshot、plain token transformer 和 next-query transformer。
- 记录 ablation 名称、core 类型、JEPA checkpoint、embedding/query 配置等 metadata，便于后续结果汇总。

**Non-Goals:**

- 不修改 `gps_conditioned_jepa` 预训练 loss、mask sampler、EMA target encoder 或 checkpoint 格式。
- 不新增 raw latitude/longitude 输入，不绕过现有 GPS-Rel-Polar `[distance, sin_theta, cos_theta]` 契约。
- 不引入蒸馏 teacher、HiST/Hist 兼容层或旧入口。
- 不把所有 fusion 配置重新整理为新目录结构；只新增本 change 所需的明确入口和引用。
- 不默认要求真实训练跑完完整实验矩阵；实现阶段只需提供 smoke/focused tests 和可运行配置。

## Decisions

1. **新增 core，而不是替换现有 `token_transformer`。**
   - 方案：在 `REPRESENTATION_CORES` 注册 `next_beam_query_transformer`，与 `single_gru`、`early_concat_gru`、`snapshot_frame`、`token_transformer` 并列。
   - 理由：现有 `token_transformer` 只是把 `[B, K, T, D]` reshape 成 token 序列再编码，当前没有显式 time/modality embedding，也没有下一时刻 query；直接替换 GRU 会让“预测下一帧”的语义不够清晰。
   - 备选：修改现有 `token_transformer`。该方案会改变已有配置语义，不适合作为低风险新增能力。

2. **使用 learned `[NEXT_BEAM]` query 产生单步输出。**
   - 方案：将历史多模态 token 与一个 learned query token 拼接后送入 Transformer Encoder，最后取 query token 输出并 reshape 为 `[B, 1, D_out]`。
   - 理由：`num_pred=1` 时，下游只需要下一时刻 beam logits，query token 能把“读取历史、预测未来一步”的意图画清楚，也避免对全部历史时刻逐一分类。
   - 备选：Transformer 输出每个时间步后取最后一帧。该方案实现简单，但仍是历史 token 表征，不如 query 形式表达预测目标。

3. **time embedding 与 modality embedding 是默认必需组件。**
   - 方案：为每个 `[modality, time]` token 加上 modality embedding 和 time embedding；query token 使用独立 query embedding，可选加入 query layer norm。
   - 理由：Transformer 本身不感知 token 顺序或模态来源；没有 embedding 的 plain transformer 应保留为 ablation，而不是主方法。
   - 备选：只使用 sinusoidal time embedding。该方案减少参数，但当前配置和 checkpoint metadata 更适合可学习 embedding，且历史长度较短。

4. **先保留 mean-pooled JEPA image feature，GPS-conditioned patch attention 作为后续增强。**
   - 方案：本 change 的下游 image branch 继续使用 `jepa_context_image` 的 patch mean pooling 输出 `[B, T, D]`。
   - 理由：用户当前要求是升级下游 GRU 融合；同时改 patch pooling 会混入第二个变量，影响 ablation 解释。
   - 备选：新增 GPS-conditioned spatial attention pooling。它有潜在收益，但应作为下一轮 change 或可选 extension。

5. **ablation 使用同一 encoder/projector/head 边界。**
   - 方案：四组配置只改变 `representation_core` 和必要的 `seq_len`/run_name/ablation metadata；JEPA checkpoint、image encoder、GPS encoder、projectors、beam head 尽量保持一致。
   - 理由：这样结果差异主要来自 fusion core，而不是 backbone 或训练 recipe。
   - 备选：为每个 ablation 单独调参。该方案可能提高单个结果，但会削弱结构对比可信度。

## Risks / Trade-offs

- **[Risk] next-query Transformer 参数更多，小数据下可能过拟合。** -> 保留 GRU 与 snapshot ablation，默认配置使用较小 `d_model=64`、少层数、dropout 和 weight decay。
- **[Risk] `snapshot_frame` 需要 `seq_len=1`，与历史模型配置不一致。** -> snapshot ablation 必须单独配置 `seq_len=1`、`num_pred=1`，并在 spec 和测试中覆盖拒绝历史时间维的现有契约。
- **[Risk] plain token transformer 与 next-query transformer 的区别不明显。** -> metadata 和配置必须明确记录 time/modality embedding、query token 和 output selection。
- **[Risk] 输出 `[B,1,D]` 可能暴露训练侧 horizon 对齐问题。** -> 使用现有 `num_pred=1` beam target 契约，并添加 forward shape/config smoke 测试。
- **[Risk] 新增配置使 `configs/fusion/` 根目录再次膨胀。** -> 新配置应放在明确的实验子目录或遵循当前配置面 guardrail；若必须放根目录，需要同步 inventory/测试。

## Migration Plan

1. 在 `modular_sequence` 中新增 `next_beam_query_transformer` core，保持现有 core 不变。
2. 新增 focused tests 验证 core shape、embedding/query 配置、错误信息和 `ModularSequenceModel` 集成。
3. 新增 JEPA downstream 配置矩阵，覆盖 GRU、snapshot、plain token transformer 和 next-query transformer。
4. 更新 run metadata/objective metadata 或配置 metadata，使 ablation 与 core 参数可追踪。
5. 运行 OpenSpec 校验和 focused tests，再根据实现触及面运行架构边界或配置加载检查。

回滚策略：删除新增 core 注册、配置和测试即可回到当前 GRU 主路径；现有 `early_concat_gru`、`token_transformer`、`snapshot_frame` 和 JEPA checkpoint 格式不应被破坏。

## Open Questions

- 新配置应放在 `configs/fusion/` 根目录还是实验子目录，需要结合当前配置面 guardrail 决定。
- next-query Transformer 的默认层数、head 数、dropout 是否沿用 `token_transformer` 默认值，还是为低数据量单独收窄。
- 是否在同一 change 中添加结果汇总脚本，统一读取四组 ablation summary；若实现面过大，可留到后续分析 workflow。
