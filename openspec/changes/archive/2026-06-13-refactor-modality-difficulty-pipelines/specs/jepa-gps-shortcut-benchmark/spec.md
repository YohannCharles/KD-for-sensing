## ADDED Requirements

### Requirement: Benchmark 复用统一 difficulty pipeline
JEPA GPS shortcut benchmark MUST 使用统一 modality difficulty pipeline 解析和应用 perturbation suites。现有 manifest 中的 GPS jitter、drift、missing/dropout、distractor、image degradation、temporal delay、sampling-rate mismatch 和 Scenario C suite type MUST 继续可解析，但 runner 内部 MUST 委托 shared difficulty operator，而不是维护独立实现分支。

#### Scenario: 旧 perturbation suite 映射到 difficulty operator
- **WHEN** benchmark manifest 使用现有 `gps_gaussian_jitter`、`image_occlusion` 或 `temporal_delay` suite type
- **THEN** runner MUST 将 suite 标准化为对应 difficulty profile/operator
- **AND** 输出 `metrics_by_condition.csv`、`robustness_summary.csv` 和 benchmark manifest 的核心列 MUST 保持兼容

#### Scenario: Scenario C preset 使用 shared GPS async operator
- **WHEN** benchmark manifest 引用 canonical Scenario C preset
- **THEN** runner MUST 通过 shared GPS async operator 构造 `C0_sync` 到 `C4_severe_async`
- **AND** metadata MUST 继续记录 max delay、GPS stride、dropout probability、fallback、source index 或等价 replay 字段

#### Scenario: benchmark 和 evaluation 使用相同扰动
- **WHEN** benchmark 与 evaluation 配置使用相同 profile id、operator、condition、severity、seed、split 和 sample id
- **THEN** 二者应用到同一 synthetic batch 时 MUST 产生一致的扰动输入、mask 和 warnings

### Requirement: Benchmark 输出 difficulty provenance
Benchmark 输出 MUST 记录 shared difficulty pipeline provenance，包括 profile id、operator registry name、resolved operator parameters、profile digest、seed 派生字段、stage、split 和 replay metadata。该 provenance MUST 与模型 comparability metadata 分开记录，避免把输入难度误当成模型结构差异。

#### Scenario: manifest 记录 difficulty provenance
- **WHEN** benchmark 完成一个 difficulty suite
- **THEN** `benchmark_manifest.json` 或等价输出 MUST 包含 difficulty profile digest、operator 列表、condition/severity、seed 和 warnings
- **AND** 模型 config、checkpoint provenance、split metadata 与 difficulty provenance MUST 分字段记录

#### Scenario: strict comparability 允许同一 difficulty profile
- **WHEN** 多个模型在同一 split、label space 和同一 difficulty profile digest 下评估
- **THEN** comparability 校验 MUST 不因共享 difficulty metadata 而失败
- **AND** 若模型使用不同 difficulty profile digest，系统 MUST 拒绝写入同一严格可比较汇总或标记为不可比较
