# jepa-next-beam-query-transformer Specification

## Purpose
记录 JEPA context image encoder 下游 next-beam query Transformer 的模型边界、配置矩阵和运行 metadata 契约，确保该路线只复用 JEPA context encoder 权重，并以可复现的 supervised beam ablation 与现有 GRU、snapshot、plain token transformer 基线比较。
## Requirements
### Requirement: JEPA 下游 next-beam query Transformer 主方法
系统 MUST 提供 JEPA context image encoder 下游复用的 next-beam query Transformer 主方法。该方法 MUST 保留 `jepa_context_image` 作为 image encoder，从 JEPA checkpoint 加载 `context_encoder` 权重，并将 image/GPS 历史表征交给 `next_beam_query_transformer` 生成单步下一时刻 beam logits。

#### Scenario: 构建 next-query JEPA downstream 模型
- **WHEN** 用户配置 `model.primary.type: modular_sequence`、启用 `image` 和 `gps`，且 image encoder 为 `jepa_context_image`
- **THEN** 系统 MUST 从配置的 JEPA checkpoint 加载 context encoder 权重
- **AND** 系统 MUST 构建 GPS MLP、projectors、`next_beam_query_transformer` core 和 beam head
- **AND** 模型 forward MUST 输出形状 `[B, 1, num_classes]` 的 beam logits

#### Scenario: next-query 主方法不重训 JEPA 预训练目标
- **WHEN** 用户运行 next-query JEPA downstream supervised beam prediction
- **THEN** 训练 MUST 使用 beam target 和 beam supervised loss
- **AND** 训练 MUST NOT 要求 JEPA latent prediction loss、target encoder EMA 更新、distiller 或外部 frozen teacher

### Requirement: JEPA downstream ablation 矩阵
系统 MUST 提供可复现的 JEPA downstream ablation 配置矩阵，用于比较 GRU、snapshot、plain token transformer 和 next-query transformer 四类 fusion core。四组 ablation MUST 尽量复用相同 JEPA checkpoint、image encoder、GPS encoder、projector、beam head、beam label space 和训练 recipe。

#### Scenario: GRU ablation
- **WHEN** 用户选择 `jepa_gru` ablation
- **THEN** 配置 MUST 使用 `early_concat_gru` representation core
- **AND** 配置 MUST 保留 JEPA context image encoder 和 GPS MLP 输入路径

#### Scenario: Snapshot ablation
- **WHEN** 用户选择 `jepa_snapshot` ablation
- **THEN** 配置 MUST 使用 `snapshot_frame` representation core
- **AND** 配置 MUST 设置与 snapshot 契约一致的 `seq_len=1` 和 `num_pred=1`

#### Scenario: Plain token transformer ablation
- **WHEN** 用户选择 `jepa_plain_token_transformer` ablation
- **THEN** 配置 MUST 使用现有 `token_transformer` representation core
- **AND** 配置 MUST NOT 声明 learned next-beam query 为启用状态

#### Scenario: Next-query transformer ablation
- **WHEN** 用户选择 `jepa_next_query_transformer` ablation
- **THEN** 配置 MUST 使用 `next_beam_query_transformer` representation core
- **AND** 配置 MUST 启用 time embedding、modality embedding 和 learned next-beam query

### Requirement: 下游运行 metadata 可追踪
JEPA downstream 运行产物 MUST 记录足够的结构 metadata，便于区分预训练来源、fusion core 和 ablation 变量。metadata MUST 至少包含 ablation 名称、representation core 类型、JEPA checkpoint 路径、是否 freeze image encoder、是否启用 time embedding、是否启用 modality embedding、是否启用 next-beam query。

#### Scenario: 写出 next-query metadata
- **WHEN** next-query JEPA downstream 训练完成并写出 `final_config.yaml` 或运行 metadata
- **THEN** metadata MUST 记录 `ablation=jepa_next_query_transformer`
- **AND** metadata MUST 记录 `representation_core.type=next_beam_query_transformer`
- **AND** metadata MUST 记录 JEPA checkpoint 路径和 query/time/modality embedding 配置

#### Scenario: 写出 ablation metadata
- **WHEN** 任一 JEPA downstream ablation 训练完成
- **THEN** metadata MUST 能区分 `jepa_gru`、`jepa_snapshot`、`jepa_plain_token_transformer` 和 `jepa_next_query_transformer`
- **AND** metadata MUST 不把 plain token transformer 标记为 next-query transformer

### Requirement: 配置与验证入口
系统 MUST 提供无副作用的配置加载和 focused test 路径，验证四组 JEPA downstream ablation 可构建、输入输出 shape 正确、且不破坏当前 beam objective 指标契约。

#### Scenario: 配置加载 smoke
- **WHEN** 开发者在 `kd_mm_beam` 环境中加载四组 JEPA downstream ablation 配置
- **THEN** 配置 MUST 解析成功
- **AND** `experiment.objective` MUST 保持为 supervised beam prediction objective 或默认 beam objective

#### Scenario: forward shape smoke
- **WHEN** 开发者使用 synthetic image/GPS batch 对四组 ablation 执行模型 forward
- **THEN** 每组输出 MUST 包含 beam logits
- **AND** next-query 与 snapshot 输出时间维 MUST 为 `1`

### Requirement: 不扩大旧研究线
JEPA next-query downstream MUST 不重新引入已退役的 HiST/Hist、KD distillation、teacher_no_kd、student_no_kd、no_kd、logits_kd 或 legacy fusion 兼容入口。

#### Scenario: 构建路径保持当前包结构
- **WHEN** 开发者实现 JEPA next-query downstream
- **THEN** 新模型能力 MUST 通过 `src/kd_sensing` 当前注册表和 `modular_sequence` 边界接入
- **AND** 实现 MUST NOT 新增绕过当前包结构的旧入口或兼容聚合层
