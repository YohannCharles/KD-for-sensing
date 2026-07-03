# beam-topology-prototype-alignment Specification

## Purpose
定义 BTAPA beam-neighborhood prototype alignment 的 soft target、prototype loss、ADBA-aware auxiliary loss、smoke validation 和本地分析边界，使 Scene31 缺失模态实验能显式利用 beam 拓扑邻接而不恢复旧 KD/runtime 入口。
## Requirements
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

### Requirement: Scene31 BTAPA local ablation workflow
项目 MUST 提供 Scene31 BTAPA local/manual ablation 配置、串行 launcher 和只读分析脚本。该 workflow MUST 使用当前 `kd-sensing-train` CLI，不得新增旧 root 训练入口。

#### Scenario: BTAPA launcher dry-run
- **WHEN** 用户运行 `conda run -n kd_mm_beam bash scripts/run_btapa_experiments.sh --dry_run --num_workers 4 --max_parallel 1`
- **THEN** launcher MUST 只打印每个 BTAPA 实验的训练命令
- **AND** 默认 `max_parallel` MUST 为 1

#### Scenario: BTAPA 输出隔离
- **WHEN** 用户运行任一 BTAPA 配置
- **THEN** 输出 run name MUST 包含 `btapa`
- **AND** 系统 MUST 不覆盖旧 `main_v3_strong_reliability_proto` 输出

### Requirement: BTAPA tau1 主候选验证分析
BTAPA 本地分析 MUST 能将 `main_v3_strong_reliability_btapa_tau1` 标记为 candidate main，并基于读取到的指标生成保守的整体结论和 paper-ready observation。结论 MUST 使用 CSV 中真实数值，不得声称未验证的显著性或最终主模型地位。

#### Scenario: candidate main 输出
- **WHEN** 用户运行 `scripts/analyze_btapa_runs.py --candidate main_v3_strong_reliability_btapa_tau1`
- **THEN** 输出 Markdown MUST 标记 candidate main
- **AND** 报告 MUST 比较 tau1、tau4、ADBA-aware、fusiononly 和 modw1 的相对表现

#### Scenario: paper-ready observation 保守生成
- **WHEN** 分析脚本能读取 proto baseline 和 BTAPA tau1 指标
- **THEN** 报告 MUST 生成一段可用于论文草稿的 observation
- **AND** observation MUST 基于读取到的 full、avg_missing 或 radar_only 数字，避免夸大

### Requirement: BTAPA tau1 多 seed mean±std
系统 MUST 提供 BTAPA tau1 多 seed 只读分析脚本，读取原始 tau1、seed2、seed3 和可选 proto/旧 V3 seed，输出 seed metrics、mean±std、Markdown 和 delta-vs-proto mean。部分 seed 尚未跑完时，脚本 MUST 继续基于已有 seed 计算并记录 n。

#### Scenario: seed 未完成仍输出
- **WHEN** seed2 或 seed3 的 metrics/checkpoint 尚不存在
- **THEN** 脚本 MUST 打印 warning 并继续
- **AND** mean±std 输出 MUST 记录实际 n，并在 Markdown 中列出 missing runs

#### Scenario: 输出核心结论
- **WHEN** 脚本生成 BTAPA tau1 与 proto baseline 的 mean±std 表
- **THEN** 末尾 MUST 打印 avg_missing Top-1、radar_only Top-1、delta mean、是否超过 proto mean 以及差异是否小于 std 的谨慎提示

### Requirement: pattern-conditional BTAPA prototype loss
系统 MUST 支持 pattern-conditional BTAPA。启用 `use_pattern_conditional_btapa=true` 时，batch 内每个 sample MUST 根据 available mask 解析 pattern name；在 `btapa_apply_patterns` 中的样本 MUST 使用 BTAPA soft beam target，其它样本 MUST 在 `btapa_fallback_to_ordinary_proto=true` 时使用 ordinary prototype target。

#### Scenario: sample-wise 混合 target
- **WHEN** 同一 batch 同时包含 `radar_only` 和 `missing_gps`
- **THEN** `radar_only` 样本 MUST 使用 BTAPA soft beam target
- **AND** `missing_gps` 样本 MUST 使用 ordinary prototype target

#### Scenario: 缺失模态不参与 modality proto loss
- **WHEN** 某样本的 available mask 中 `radar=0`
- **THEN** radar modality feature MUST 不参与 modality prototype loss
- **AND** fusion feature MUST 继续参与 prototype loss

#### Scenario: diagnostics 记录 active ratio
- **WHEN** pattern-conditional BTAPA 启用
- **THEN** 训练 metrics MUST 记录 `ordinary_proto_loss`、`btapa_loss`、`btapa_active_ratio` 和 `total_proto_loss`

