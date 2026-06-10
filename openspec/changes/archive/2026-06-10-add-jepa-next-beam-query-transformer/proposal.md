## Why

当前 JEPA 下游波束预测保留了高质量的 `jepa_context_image` 图像编码器，但融合阶段仍以 `early_concat_gru` 为主：每个时间步把 image/GPS 表征拼接后交给 GRU。对于 `num_pred=1` 的下一时刻 beam prediction，这种“全时序逐步输出”并不是最贴合任务语义的结构，也不利于明确建模模态 token、时间 token 和“下一时刻查询”之间的关系。

本变更希望在不重做 JEPA 预训练、不改变数据契约的前提下，新增一个更适合下一时刻预测的下游融合主方法：保留 JEPA image encoder，将 GRU 融合升级为带 time embedding、modality embedding 和 learned next-beam query 的 Transformer，并同时保留 GRU、snapshot 和 plain token transformer 作为可复现实验消融。

## What Changes

- 新增 JEPA downstream 主方法：`jepa_context_image` + GPS MLP + projectors + `next_beam_query_transformer` + beam head。
- 新增 `next_beam_query_transformer` representation core：
  - 接收多模态历史 token `[B, K, T, D]`。
  - 为模态和时间注入可学习 embedding。
  - 添加 learned `[NEXT_BEAM]` query token。
  - 输出单步下一时刻表征 `[B, 1, D_out]`，供现有 beam head 生成 `[B, 1, num_classes]` logits。
- 保留并配置四组 JEPA downstream ablation：
  - `jepa_gru`: 当前 `early_concat_gru` 基线。
  - `jepa_snapshot`: 当前帧/单步 `snapshot_frame` 无历史基线。
  - `jepa_plain_token_transformer`: 现有 `token_transformer` 的简单替换基线。
  - `jepa_next_query_transformer`: 新增主方法。
- 增加配置、测试和 metadata，使实验可区分 core 类型、JEPA checkpoint、mask 预训练来源、是否使用 next query、是否使用 time/modality embedding。
- 不新增数据集字段，不改变 JEPA 预训练目标，不改变 beam label space，不引入外部 frozen teacher 或蒸馏路径。

## Capabilities

### New Capabilities

- `jepa-next-beam-query-transformer`: 定义 JEPA context image encoder 下游复用时的 next-beam query Transformer 主方法、ablation 矩阵、配置和运行产物契约。

### Modified Capabilities

- `modular-sequence-model`: 增加 `next_beam_query_transformer` representation core 的输入、embedding、query、输出和 head 兼容需求。

## Impact

- 影响模型代码：`src/kd_sensing/models/modular.py` 中新增 representation core，并复用现有 encoder/projector/head 边界。
- 影响配置：新增或调整 JEPA downstream fusion 配置，覆盖 GRU、snapshot、plain token transformer 和 next-query transformer 四组 ablation。
- 影响测试：新增核心 shape/contract 测试、配置加载测试、JEPA context encoder 复用测试和 ablation 配置 smoke。
- 影响运行 metadata：记录 representation core 类型、query transformer 参数、time/modality embedding 启用状态、JEPA checkpoint 路径和 ablation 名称。
- 影响文档/OpenSpec：新增 capability spec，并修改 `modular-sequence-model` 需求。
- 所有项目相关 Python 验证继续使用 `conda run -n kd_mm_beam ...`。
