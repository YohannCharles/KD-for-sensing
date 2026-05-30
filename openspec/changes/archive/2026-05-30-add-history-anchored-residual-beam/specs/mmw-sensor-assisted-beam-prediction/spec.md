## ADDED Requirements

### Requirement: History-anchored profile 边界
MMW sensor-assisted beam prediction MUST 将默认三主模态 profile 与 history-anchored profile 清晰分离。默认 sensor-assisted 主配置 MUST 继续只使用 image、GPS 和 LiDAR 作为 sensing input；radar MUST NOT 进入默认主结论 profile，只有显式启用 history-anchored profile 时，模型才可使用历史 beam 输入。

#### Scenario: 默认 sensor-assisted 不消费 input beam
- **WHEN** 用户运行 `sensor_assisted_quick_validation` 或等价默认 sensor-assisted 配置
- **THEN** enabled sensing modalities MUST 只包含 `image`、`gps` 和 `lidar`
- **AND** model forward kwargs MUST NOT 包含 `radar_batch`
- **AND** model forward kwargs MUST NOT 包含 `input_beam_batch`、`last_beam_batch` 或等价历史 beam 输入
- **AND** summary MUST 继续把 last-beam 记录为 diagnostic baseline

#### Scenario: 显式 history-anchored run 标记 profile
- **WHEN** 用户显式设置 history-anchored profile 或 `hist_beam.history_anchor.enabled=true`
- **THEN** run metadata MUST 记录 `profile=history_anchored` 或等价机器可读字段
- **AND** run metadata MUST 记录 `uses_input_beam_as_model_input=true`
- **AND** summary MUST NOT 将该 run 静默归入默认 sensor-assisted 主结论集合

#### Scenario: profile 混用时失败或降级为不可用主结论
- **WHEN** 一个 run 同时声称默认 sensor-assisted 主结论且模型实际消费历史 beam 输入
- **THEN** 系统 MUST 将该 run 标记为 `main_conclusion_eligible=false` 或直接失败
- **AND** exclusion reason MUST 包含 `uses_input_beam_as_model_input`

### Requirement: History-anchored summary eligibility
MMW sensor-assisted summary MUST 支持过滤和比较 history-anchored run。summary MUST 同时保留默认 sensor-assisted eligibility 字段和 history-anchored residual 专用字段，使报告能够区分严格传感器输入结论与历史锚定 beam prediction 结论。

#### Scenario: summary 输出 history anchor 字段
- **WHEN** summary 汇总 history-anchored run
- **THEN** 每个 run record MUST 包含 `history_anchor_enabled`、`history_anchor_mode`、`residual_target_enabled`、`uses_input_beam_as_model_input` 和 `main_conclusion_profile`
- **AND** record MUST 保留 enabled sensing modalities 和 sensitive usage flags

#### Scenario: 默认主结论过滤排除 history-anchored run
- **WHEN** downstream report 过滤默认 sensor-assisted 主结论 run
- **THEN** 使用历史 beam 输入的 run MUST 被排除
- **AND** summary MUST 提供单独字段或集合用于 history-anchored 结论比较

#### Scenario: history-anchored 内部可比较
- **WHEN** 同一 source、target、budget 和 seed 下存在 residual-only、residual+private calibration、last-beam baseline 和 Markov delta baseline
- **THEN** summary MUST 输出这些 run/baseline 的 Top-K、NRP 和 dB loss 对比
- **AND** summary MUST 标明它们属于 history-anchored profile 内部比较

### Requirement: History beam 与 sensitive 字段审计
history-anchored profile MUST 将历史 beam 输入与 target sensitive supervision 分开审计。使用样本历史窗口中的 `input_beam` 作为模型输入不等价于使用 target future label 训练；但任何使用 target beam_power、path/radio/channel-derived label 的训练路径仍 MUST 按既有 sensitive usage 规则记录和判定。

#### Scenario: 历史 beam 输入单独记录
- **WHEN** history-anchored run 使用 `input_beam` 作为模型输入
- **THEN** run metadata MUST 记录 `used_input_beam_as_input=true`
- **AND** metadata MUST 区分 `used_target_beam_for_supervised_loss` 与 `used_input_beam_as_input`

#### Scenario: target future label 仅用于合法监督
- **WHEN** `label_budget>0` 的 history-anchored target adaptation 读取 labeled target_adapt future beam 计算 residual supervised loss
- **THEN** metadata MUST 记录 `used_target_beam_for_supervised_loss=true`
- **AND** target_test future beam MUST NOT 被训练、阈值选择或 prototype update 使用

#### Scenario: path/radio sensitive 规则继续生效
- **WHEN** history-anchored run 使用 target path、radio、beam_power 或 channel-derived 字段计算训练 loss
- **THEN** metadata MUST 记录对应 target sensitive usage flag
- **AND** 若默认 sensor-assisted 规则不允许该监督进入主结论，summary MUST 标记不可用于对应主结论
