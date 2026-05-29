## ADDED Requirements

### Requirement: 数据集无关的 LOSO fold 规划
LOSO workflow MUST 在现有 DeepSense6G 31-34 folds 之外支持数据集无关的 fold 规划。对于 MMW，planner MUST 使用 dataset descriptor 和数据可用性 metadata 生成 scenario-level、town-level 或 condition-level source/target folds。

#### Scenario: 生成 MMW scenario fold
- **WHEN** MMW data availability metadata 在请求范围内包含至少两个 ready scenarios
- **THEN** planner MUST 为每个 target scenario 生成一个 fold
- **AND** 每个 fold MUST 记录 dataset family `MMW`、condition、town、target scenario、source scenarios 和 fold id

#### Scenario: 保留 DeepSense6G 默认 folds
- **WHEN** 用户请求现有 DeepSense6G 31-34 LOSO workflow
- **THEN** planner MUST 继续生成四个现有 DeepSense6G folds
- **AND** MMW-specific metadata MUST NOT be required

### Requirement: Single-scene smoke is not LOSO
LOSO workflow MUST 区分 single-scene smoke runs 和 cross-scene adaptation runs。单个 ready MMW scenario MUST NOT 被报告为 LOSO、leave-one-scene-out、cross-town 或 cross-condition evidence。

#### Scenario: MMW 只有一个 ready scenario
- **WHEN** planner sees exactly one ready MMW scenario
- **THEN** planner MUST generate at most smoke or within-scenario sanity runs
- **AND** execution summary MUST mark `cross_scene_claim_allowed: false`

### Requirement: MMW target adapt/test no leakage
对于 MMW folds，target_adapt 和 target_test split MUST 确定性且无泄漏。Split metadata MUST 包含 sample ids、可用时的 sequence ids、scenario/town/condition、split seed、split ratio 和分布摘要。

#### Scenario: target split 无交集
- **WHEN** MMW target split is built
- **THEN** target_adapt and target_test sample ids MUST be disjoint
- **AND** if sequence segment ids are available, the two splits MUST also avoid segment overlap

#### Scenario: target_test 不参与 adaptation 决策
- **WHEN** MMW target adaptation runs
- **THEN** target_test samples MUST NOT be used for supervised loss, prototype selection, threshold tuning, normalizer fitting or early stopping
- **AND** run metadata MUST record this split boundary

### Requirement: MMW few-shot sampling strategy
LOSO workflow MUST 支持 MMW few-shot target sampling，budgets 为 `0`、`5`、`10`、`20` 和 `50`。Sampling MUST 优先覆盖 coarse sector 和 relative azimuth bins；只有当 bin 不可用或样本不足时，才退化为确定性随机采样。

#### Scenario: 分层采样成功
- **WHEN** target_adapt contains multiple coarse sectors and relative azimuth bins
- **THEN** sampler MUST select labeled samples to cover as many sector/bin combinations as possible within the budget
- **AND** sampling manifest MUST record sector/bin for every labeled sample

#### Scenario: 分层字段不可用
- **WHEN** relative azimuth or coarse sector is unavailable for target_adapt samples
- **THEN** sampler MUST fall back to deterministic random sampling
- **AND** sampling manifest MUST record the fallback reason

### Requirement: MMW LOSO summary claim guard
MMW LOSO summary MUST 包含机器可读 claim guard，用于说明某个 run 是否能支撑 cross-scene、cross-town 或 cross-condition 结论。

#### Scenario: summary 输出 claim guard
- **WHEN** MMW smoke, LOSO or adaptation execution completes
- **THEN** summary MUST include `claim_scope` and `cross_scene_claim_allowed`
- **AND** incomplete or single-scene runs MUST set `cross_scene_claim_allowed` to false
