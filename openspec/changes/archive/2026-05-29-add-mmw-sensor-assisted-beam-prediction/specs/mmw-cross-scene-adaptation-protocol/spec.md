## ADDED Requirements

### Requirement: MMW sensor-assisted quick validation protocol
MMW cross-scene adaptation protocol MUST allow a sensor-assisted quick validation mode for rapid iteration. This mode MUST be machine-readable in plan metadata and MUST restrict the default matrix to `label_budget=10` and two seeds.

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

#### Scenario: modality profile 可审计
- **WHEN** plan、run metadata 或 summary 写出
- **THEN** metadata MUST include enabled sensing modalities and excluded radio/channel/path fields
- **AND** metadata MUST show whether LiDAR/radar derived cache was used
- **AND** metadata MUST allow downstream reports to filter sensor-assisted runs separately from mmWave-assisted runs
