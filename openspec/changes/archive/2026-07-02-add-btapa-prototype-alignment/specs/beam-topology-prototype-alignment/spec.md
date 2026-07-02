## ADDED Requirements

### Requirement: Beam topology soft prototype target
系统 MUST 支持 BTAPA beam-neighborhood soft target。该 target MUST 基于 hard beam label、`num_beams`、`tau_beam` 和可配置 circular distance 生成 `[B, num_beams]` 概率分布，且每行概率和 MUST 在数值容差内等于 1。

#### Scenario: 非环形 beam distance
- **WHEN** `proto_target_type=beam_soft` 且 `circular_beam_distance=false`
- **THEN** target MUST 使用 `abs(k-y)` 作为 beam distance
- **AND** target MUST 为 finite 且 shape 为 `[B, num_beams]`

#### Scenario: 环形 beam distance
- **WHEN** `proto_target_type=beam_soft` 且 `circular_beam_distance=true`
- **THEN** target MUST 使用 `min(abs(k-y), num_beams-abs(k-y))` 作为 beam distance
- **AND** beam 0 与最后一个 beam MUST 可作为邻近 beam

### Requirement: BTAPA prototype loss composition
系统 MUST 使用 `BeamPrototypeBank` 对 fusion feature 和可选可用 modality feature 计算 cosine prototype logits，并用 soft CE 计算 BTAPA loss。`lambda_proto` MUST 作为总 prototype loss 权重，fusion/modality 权重 MUST 由配置控制。

#### Scenario: fusion feature 参与 prototype loss
- **WHEN** `use_beam_topology_proto=true` 且 `btapa_include_fusion=true`
- **THEN** fusion feature MUST 参与 prototype loss
- **AND** diagnostics MUST 记录 `btapa_fusion_loss`

#### Scenario: 缺失 modality 不参与 prototype loss
- **WHEN** 提供 `modality_features: [B, M, D]` 和 `mask: [B, M]`
- **THEN** 只有 `mask==1` 的 modality feature 能参与 modality prototype loss
- **AND** 缺失 modality MUST 不贡献 modality prototype sample count

### Requirement: ADBA-aware auxiliary prototype loss
系统 MUST 提供默认关闭的 ADBA-aware auxiliary prototype loss。该 loss MUST 基于 beam distance 构造 near-beam mask，并最小化 `-log(sum softmax(proto_logits)[near_mask])`。

#### Scenario: ADBA-aware loss 默认关闭
- **WHEN** `use_adba_aware_proto=false` 或未声明
- **THEN** 系统 MUST 记录 `adba_proto_loss` 为 0 或空值
- **AND** 总训练 loss MUST 不包含 ADBA-aware prototype loss

#### Scenario: ADBA-aware loss 启用
- **WHEN** `use_adba_aware_proto=true`
- **THEN** 系统 MUST 使用 `adba_margin` 和 circular distance 配置计算 near-beam mask
- **AND** 总训练 loss MUST 加入 `lambda_adba_proto * L_adba_proto`

### Requirement: BTAPA smoke validation
系统 MUST 提供无需真实 dataset 的 BTAPA smoke test，用 synthetic feature、label 和 available mask 验证 loss finite、target 归一化、缺失 modality mask、circular/non-circular 路径和 backward。

#### Scenario: synthetic BTAPA loss backward
- **WHEN** 运行 `conda run -n kd_mm_beam python scripts/smoke_test_btapa.py`
- **THEN** smoke test MUST 完成 BTAPA forward 和 backward
- **AND** 不得读取真实 `dataset/` 或写入 tracked runtime artifact

### Requirement: BTAPA local analysis
系统 MUST 提供只读分析脚本比较旧 V3 与 BTAPA 消融 missing-pattern 指标，并写出 CSV/Markdown 汇总到 `outputs/scene31/analysis/`。

#### Scenario: 比较 V3 与 BTAPA
- **WHEN** 用户运行 `scripts/analyze_btapa_runs.py`
- **THEN** 脚本 MUST 读取 baseline 和 BTAPA runs 的 missing-pattern CSV/JSON
- **AND** 输出 comparison、delta-vs-baseline 和文本结论
