# research-claim-harvester Specification

## Purpose
定义只读 research claim harvester、strict comparability gate、experiment ledger 和 dashboard 输出，用于把本地运行产物整理为 candidate-only 研究证据。
## Requirements
### Requirement: Research claim harvesting
系统 MUST 提供只读 research claim harvester，用于从本地训练、评估和诊断产物中生成 claim candidate。Harvester MUST 不移动、删除、重写或压缩任何 `outputs/`、`logs/`、checkpoint、cache 或真实数据。

#### Scenario: 收割 Scene31 missing-pattern 结果
- **WHEN** 用户对 `outputs/scene31/eval/` 或等价输出目录运行 harvester
- **THEN** 系统 MUST 发现 `*_missing_patterns.csv` 和 `*_missing_patterns.json`
- **AND** 输出 MUST 记录 run name、method、seed、pattern、metric fields、sample count、source artifact path 和 generated_at
- **AND** 缺少 seed、split、metric profile 或 label space 时 MUST 将 candidate 标记为 `needs_review` 或 `not_comparable`

#### Scenario: 收割训练 run metrics
- **WHEN** harvester 扫描包含 `metrics.csv`、`metrics.json`、`train_log.json`、`final_config.yaml` 或 checkpoint sidecar 的 run 目录
- **THEN** 系统 MUST 抽取 run identity、config path、config digest、seed、scene scope、selection metric、best checkpoint path 和主要验证/测试指标
- **AND** 无法解析的字段 MUST 作为 warning 输出，而不是导致整个收割失败

#### Scenario: 只生成候选 claim
- **WHEN** harvester 发现完整且可比的结果
- **THEN** 系统 MUST 生成 `claim_candidate` 或等价草稿记录
- **AND** 该记录 MUST 包含 `candidate_only=true` 或等价字段
- **AND** 系统 MUST NOT 自动修改 `docs/result_claims_registry.md`

### Requirement: Strict comparability gate
Harvester MUST 对 claim candidate 执行 strict comparability gate。Gate MUST 至少检查 split、sample_count、label_space、metric_profile、target_source、difficulty_digest、seed 和 config/run family。

#### Scenario: 可比字段完整
- **WHEN** 同一 method group 的多 seed 结果具有相同 split、label space、metric profile、target source 和 difficulty digest
- **THEN** candidate MUST 标记 comparability 为 `strict`
- **AND** candidate MAY 进入 claim draft 输出

#### Scenario: 可比字段缺失或冲突
- **WHEN** 结果之间的 split、sample_count、label_space、metric_profile、target_source 或 difficulty_digest 不一致
- **THEN** candidate MUST 标记为 `not_comparable`
- **AND** warning MUST 记录冲突字段、期望值和实际值

### Requirement: Experiment ledger
系统 MUST 支持轻量 experiment ledger，用于保存 harvested run/candidate 的机器可读历史。默认 ledger 输出 MUST 位于 ignored `outputs/analysis/research_ledger/` 或用户显式指定路径。

#### Scenario: 写出 JSONL ledger
- **WHEN** 用户启用 ledger 输出
- **THEN** 系统 MUST 写出 JSONL、CSV 或 SQLite 中至少一种机器可读格式
- **AND** 每条记录 MUST 包含 run_id、run_name、config_path、config_digest、seed、scene_scope、artifact_paths、metric_summary、claim_status、comparability_status 和 caveat

#### Scenario: ledger 不提交运行产物
- **WHEN** ledger 引用 checkpoint、metrics、figures、logs 或 cache
- **THEN** ledger MUST 只记录路径、摘要、mtime、size 或 digest
- **AND** 系统 MUST NOT 把真实 checkpoint、metrics CSV、figures、cache 或 logs 纳入源码变更

### Requirement: Daily research dashboard
系统 MUST 提供 daily research dashboard，用于给研究者快速查看当前实验状态、资源状态和 claim readiness。Dashboard MUST 可输出人类可读摘要和机器可读 JSON。

#### Scenario: dashboard 聚合运行和 claim 状态
- **WHEN** 用户运行 dashboard 命令
- **THEN** 输出 MUST 至少包含 active OpenSpec change 摘要、running/waiting/failed/stale run 计数、GPU/进程快照、pending/unverified claim 计数、可升级 claim candidate 和缺失的下一步
- **AND** 命令 MUST 保持只读

#### Scenario: dashboard 输出 next action
- **WHEN** 某个 method 缺少 seed、fresh eval、strict field 或 checkpoint provenance
- **THEN** dashboard SHOULD 输出 next-action hint
- **AND** hint MUST 不自动启动训练、评估、清理或文档修改
