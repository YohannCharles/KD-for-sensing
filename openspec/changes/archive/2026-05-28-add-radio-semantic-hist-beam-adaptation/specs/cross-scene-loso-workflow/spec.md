## ADDED Requirements

### Requirement: Radio-semantic few-shot sampling
LOSO workflow MUST support radio-semantic-aware target labeled subset sampling. When radio-semantic labels are legally available for target_adapt labeled sampling, sampler MUST prioritize radio-semantic stratification, then coarse sector and relative azimuth bin stratification, then deterministic random fallback.

#### Scenario: radio-semantic 分层采样
- **WHEN** `label_budget` 大于 0 且 target_adapt 样本包含合法 `radio_semantic_label`
- **THEN** sampler MUST 优先选择覆盖不同 radio-semantic classes 的 labeled samples
- **AND** sampling manifest MUST 记录每个 labeled sample 的 radio label、beam、coarse sector、relative azimuth bin、seed 和 label source

#### Scenario: radio label 不可用时退化
- **WHEN** target_adapt 缺少合法 radio labels 但存在 coarse sector 或 relative azimuth bin
- **THEN** sampler MUST 退化为 coarse/azimuth 分层采样
- **AND** sampling metadata MUST 记录 radio stratification unavailable reason

### Requirement: Radio-semantic target 防泄漏
LOSO execute MUST enforce radio-semantic leakage boundaries during target adaptation. For `label_budget=0` and unlabeled target_adapt batches, target beam labels, beam_power, q_power and radio_semantic_label MUST NOT be used as supervised training targets or prototype labels.

#### Scenario: 0-label run 记录未使用 target radio label
- **WHEN** LOSO runner 执行 radio-semantic variant 且 `label_budget=0`
- **THEN** adaptation metadata MUST 记录 `used_target_labels=false`
- **AND** metadata MUST 记录 `used_target_beam_power_for_training=false`
- **AND** metadata MUST 记录 `used_target_radio_label_for_training=false`

#### Scenario: target_test 不参与 radio prototype 更新
- **WHEN** target_test evaluation 包含 beam_power 或 radio labels
- **THEN** runner MUST 只将这些字段用于离线 metrics
- **AND** runner MUST NOT 使用 target_test 字段更新 radio prototypes、target-private prototypes、confidence threshold 或 early stopping

### Requirement: Radio-semantic quick validation conclusion
LOSO summary and quick validation conclusion MUST compare coarse prototype and radio-semantic prototype variants with enough diagnostics to judge whether radio semantics contributed beyond adapter-only and coarse prototype baselines.

#### Scenario: V5 vs V6 对比
- **WHEN** 同一 fold、budget 和 seed 下存在 V5 coarse prototype 与 V6 radio prototype metrics
- **THEN** conclusion MUST 比较 Top-1/3/5、coarse accuracy、radio accuracy、power metrics、prototype coverage、trainable ratio 和 adaptation time
- **AND** conclusion MUST 标明 radio prototype 是否优于 coarse prototype

#### Scenario: radio condition off/on 对比
- **WHEN** 同一 fold、budget 和 seed 下存在 radio condition off 与 on 的 V6 runs
- **THEN** conclusion MUST 比较 beam metrics 与 radio assignment diagnostics
- **AND** 若 on/off prediction 完全一致，conclusion MUST 记录 `radio_condition_prediction_delta=0` 或等价诊断

#### Scenario: 缺失 radio 指标时不可判定
- **WHEN** 生成 radio conclusion 所需的 radio label、beam_power、prototype artifact 或 metrics 缺失
- **THEN** conclusion MUST 将对应比较标记为 `inconclusive`
- **AND** conclusion MUST 记录缺失字段和 run path
