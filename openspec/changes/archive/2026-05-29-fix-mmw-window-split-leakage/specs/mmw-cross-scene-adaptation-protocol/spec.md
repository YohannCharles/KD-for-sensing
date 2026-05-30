## MODIFIED Requirements

### Requirement: MMW scenario/town/weather split
系统 MUST 支持基于 MMW 数据可用性登记生成 scenario-level、town-level 和 weather/condition-level source-target split。split MUST 保留 target_adapt/target_test 防泄漏约束，并 MUST 支持单场景 smoke、scenario-LOSO、leave-one-town-out 和 leave-one-condition-out 四类 protocol。对于基于滑窗的 MMW sequence 数据，source、target_adapt 和 target_test 的隔离 MUST 不只检查 sample id，还 MUST 检查时间窗口上下文、frame overlap 和 guard band，避免相邻窗口跨 split 泄漏。

#### Scenario: 单场景只生成 smoke split
- **WHEN** 可用 MMW scenario 少于两个
- **THEN** planner MUST 只生成 single-scene smoke 或 within-scenario sanity split
- **AND** plan metadata MUST 标明该 split 不可用于跨场景结论

#### Scenario: scenario-LOSO split
- **WHEN** 同一 town/condition 下至少两个 scenario 可用
- **THEN** planner MUST 能选择一个 scenario 作为 target，其余 scenario 作为 source
- **AND** target scenario MUST 被确定性拆分为 target_adapt 和 target_test
- **AND** source、target_adapt 和 target_test 的 sample id MUST 无交集
- **AND** target_adapt 与 target_test MUST 使用 group-safe 或等价防泄漏协议，避免共享完整历史+未来窗口 frame id
- **AND** split metadata MUST 记录 target_adapt/target_test 的 guard band、group key、window overlap diagnostics 和 strict eligibility

#### Scenario: condition-level split
- **WHEN** sunny、rainy、foggy 或其它 condition 中至少两个 condition 可用
- **THEN** planner MUST 能将一个 condition 作为 target condition
- **AND** source condition 与 target condition MUST 在 metadata 中明确记录

### Requirement: MMW sensor-assisted quick validation protocol
MMW cross-scene adaptation protocol MUST allow a sensor-assisted quick validation mode for rapid iteration. This mode MUST be machine-readable in plan metadata and MUST restrict the default matrix to `label_budget=10` and two seeds. Quick validation MUST consume split eligibility metadata; unknown 或高重叠 split 只能作为 debug/sanity 运行，不得作为主结论。

#### Scenario: quick validation matrix metadata
- **WHEN** planner builds an MMW sensor-assisted quick validation plan
- **THEN** plan metadata MUST include `profile=sensor_assisted_quick_validation` or equivalent machine-readable marker
- **AND** plan metadata MUST include `budgets=[10]`
- **AND** plan metadata MUST include exactly two seeds unless explicitly overridden
- **AND** plan metadata MUST record that results are quick validation rather than full budget sweep

#### Scenario: scenario-LOSO remains the only claim with current data
- **WHEN** local MMW availability contains only sunny/Town10 ready scenarios
- **THEN** sensor-assisted quick validation MAY produce scenario-LOSO conclusions
- **AND** it MUST NOT claim leave-one-town-out or weather-shift validation
- **AND** summary MUST record unavailable protocol scope when town/weather data is missing

#### Scenario: source-target split 防泄漏
- **WHEN** sensor-assisted quick validation samples target labeled subset with `label_budget=10`
- **THEN** selected labeled sample ids MUST come only from target_adapt
- **AND** target_test sample ids MUST remain disjoint from source and target_adapt
- **AND** target_test labels, beam_power, path fields and radio labels MUST NOT be used for adaptation threshold selection or training loss
- **AND** target_adapt labeled samples MUST NOT share full or guard-band-protected sliding-window context with target_test samples
- **AND** plan or split metadata MUST expose diagnostics proving the target_adapt/target_test split is strict-validation eligible

#### Scenario: modality profile 可审计
- **WHEN** plan、run metadata 或 summary 写出
- **THEN** metadata MUST include enabled sensing modalities and excluded radio/channel/path fields
- **AND** metadata MUST show whether LiDAR/radar derived cache was used
- **AND** metadata MUST allow downstream reports to filter sensor-assisted runs separately from mmWave-assisted runs

#### Scenario: strict-ineligible split quick validation is excluded
- **WHEN** quick validation run 使用 `strict_validation_eligible=false` 的 split metadata
- **THEN** run metadata 和 summary MUST 标记该 run 不可用于主结论
- **AND** summary MUST 将该 run 归入 debug/sanity 或 excluded results
- **AND** exclusion reason MUST 包含 split strategy 或 leakage diagnostics 中的机器可读原因
