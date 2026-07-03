## ADDED Requirements

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
