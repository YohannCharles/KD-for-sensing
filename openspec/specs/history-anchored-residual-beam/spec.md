# history-anchored-residual-beam Specification

## Purpose
TBD - created by archiving change add-history-anchored-residual-beam. Update Purpose after archive.
## Requirements
### Requirement: History-anchored beam 输入契约
系统 MUST 支持一个显式 opt-in 的 history-anchored beam prediction profile。启用该 profile 后，模型训练和评估 MUST 能消费样本历史窗口中的 `input_beam` 或等价 beam history；未启用该 profile 时，现有 sensor-assisted 和普通 HiST-Beam 输入语义 MUST 保持不变。

#### Scenario: 默认不启用历史 beam 输入
- **WHEN** 用户运行未设置 `hist_beam.history_anchor.enabled=true` 的现有 HiST-Beam、P3 或 sensor-assisted 配置
- **THEN** batch preparation MUST NOT 将 `input_beam` 传入模型 forward
- **AND** run metadata MUST NOT 声称模型使用了历史 beam anchor

#### Scenario: 显式启用历史 beam 输入
- **WHEN** 用户设置 `hist_beam.history_anchor.enabled=true`
- **THEN** batch preparation MUST 从样本历史窗口构造 `input_beam_batch` 或等价 `last_beam` tensor
- **AND** 模型 forward MUST 接收该历史 beam 输入并将其用于 history-anchored prediction
- **AND** run metadata MUST 记录 `uses_input_beam_as_model_input=true`

#### Scenario: 历史 beam 只来自样本历史窗口
- **WHEN** 系统构造 history-anchored batch
- **THEN** `input_beam_batch` MUST 只来自该样本预测时刻之前的历史 beam 字段
- **AND** 系统 MUST NOT 使用 target future beam、target_test label、beam_power argmax 或 path/radio derived label 构造模型输入

### Requirement: Circular residual beam label
history-anchored residual 模式 MUST 将 beam 预测目标定义为相对最后历史 beam 的环形 residual/delta。系统 MUST 支持 `num_classes=64` 的默认 beam codebook，并 MUST 对任意合法 `num_classes` 使用一致的 modulo 语义。

#### Scenario: 计算单 horizon residual label
- **WHEN** `last_beam=62`、`future_beam=1` 且 `num_classes=64`
- **THEN** residual label MUST 等于 `(1 - 62) mod 64 = 3`
- **AND** residual label MUST 位于 `[0, 64)` 范围内

#### Scenario: 计算多 horizon residual label
- **WHEN** batch 中存在形状为 `[B, H]` 的 future beam label
- **THEN** 系统 MUST 生成形状为 `[B, H]` 的 residual label
- **AND** 每个 horizon 的 residual MUST 使用同一样本的最后一个历史 beam 作为 anchor，除非配置显式指定其它历史锚定策略

#### Scenario: 非法 beam label 清晰失败
- **WHEN** `last_beam` 或 `future_beam` 缺失、越界或不是整数 beam label
- **THEN** residual label builder MUST 抛出包含字段名和 sample id 的清晰错误
- **AND** 系统 MUST NOT 静默回退到绝对 beam CE

### Requirement: Residual logits 到绝对 beam 空间重建
history-anchored residual evaluation MUST 将 residual logits 环形平移回绝对 beam logits，然后复用现有 beam Top-K、power 和 prediction artifact 流程。重建 MUST 保持样本级 `last_beam` 差异和 horizon 维度。

#### Scenario: argmax residual 重建为绝对 beam
- **WHEN** 某个样本 `last_beam=62` 且 residual top-1 为 `3`
- **THEN** reconstructed absolute top-1 MUST 等于 `(62 + 3) mod 64 = 1`
- **AND** predictions artifact MUST 记录 residual top-1 和 reconstructed absolute top-1

#### Scenario: residual top-k 重建保持排序
- **WHEN** residual logits 的 top-k delta 为 `[0, 1, 63]`
- **THEN** reconstructed absolute top-k MUST 按相同 logit 排序映射为 `[(last_beam+0) mod C, (last_beam+1) mod C, (last_beam+63) mod C]`
- **AND** Top-1、Top-3、Top-5 MUST 基于 reconstructed absolute beam 与 true future beam 计算

#### Scenario: beam power 指标使用绝对预测
- **WHEN** target_test 样本包含 beam power vector
- **THEN** normalized received power 和 beam power loss dB MUST 使用 reconstructed absolute beam prediction 计算
- **AND** residual delta 本身 MUST NOT 被当作 beam power index

### Requirement: History-anchored baseline diagnostics
history-anchored summary MUST 输出足以判断历史锚定是否解决 absolute-ID 迁移崩溃的诊断 baseline。诊断 MUST 至少包含 last-beam baseline、Markov delta baseline、absolute source-only baseline 和 residual model 指标。

#### Scenario: 输出 last-beam baseline
- **WHEN** evaluation batch 存在 `input_beam`
- **THEN** metrics MUST 包含 last-beam Top-1、Top-3 或不可用原因
- **AND** summary MUST 标明 last-beam baseline 是否仅用于诊断或作为 history-anchored 可比较 baseline

#### Scenario: 输出 Markov delta baseline
- **WHEN** source train 或 labeled target_adapt split 可用于估计 delta transition
- **THEN** 系统 MUST 输出 Markov delta baseline 的 Top-K 指标
- **AND** metrics MUST 记录 Markov baseline 使用的数据 split、样本数和 smoothing 配置

#### Scenario: 诊断 source prior collapse
- **WHEN** source-only absolute model 在 target_test 的预测分布集中于 source train 主 beam
- **THEN** summary MUST 输出 predicted beam histogram、target true beam histogram 和 source train beam histogram
- **AND** summary MUST 能标记 absolute-ID prior collapse 或等价负迁移诊断

### Requirement: History-anchored 实验矩阵
系统 MUST 提供最小 history-anchored quick validation 矩阵，用于在不扩大完整 sweep 的前提下判断 residual formulation 是否有效。矩阵 MUST 支持一个 source 场景泛化到其它两个 target 场景、两个 seed 和 `label_budget=10`。

#### Scenario: 最小矩阵覆盖 residual 对比
- **WHEN** 用户运行 history-anchored quick validation 配置
- **THEN** plan MUST 覆盖 absolute source-only、history absolute classifier、residual-only 和 residual+private calibration 变体
- **AND** plan MUST 包含两个 seed 和 `label_budget=10`
- **AND** plan metadata MUST 记录该矩阵是 quick validation，不等价于完整 budget/seed sweep

#### Scenario: 不混入默认 sensor-assisted 矩阵
- **WHEN** 系统生成默认 sensor-assisted quick validation plan
- **THEN** history-anchored residual 变体 MUST NOT 被静默加入默认 plan
- **AND** 只有用户显式选择 history-anchored profile 时才生成这些 run

