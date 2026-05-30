## ADDED Requirements

### Requirement: History beam split 防泄漏
MMW cross-scene adaptation protocol MUST 在启用 history-anchored prediction 时审计历史 beam 窗口来源。`input_beam` MUST 只来自样本自身预测时刻之前的历史窗口；target future beam、target_test label、beam_power argmax 或任何由 future/channel/path/radio 派生的目标字段 MUST NOT 被写入模型历史输入。

#### Scenario: target_adapt 与 target_test 历史窗口隔离
- **WHEN** planner 生成启用 history-anchored profile 的 scenario-LOSO split
- **THEN** split metadata MUST 检查 target_adapt 和 target_test 不共享受 guard-band 保护的完整预测窗口
- **AND** overlap diagnostics MUST 覆盖用于 `input_beam` 的历史 frame 和用于 future label 的预测 frame
- **AND** strict validation eligibility MUST 反映该检查结果

#### Scenario: input beam 来自历史字段
- **WHEN** dataset 为 history-anchored run 返回 `input_beam`
- **THEN** `input_beam` MUST 来自历史 beam 字段或历史 beam label cache
- **AND** `input_beam` MUST NOT 从当前 future target beam、beam_power vector argmax 或 target_test diagnostic label 反推

#### Scenario: 历史窗口不可审计时排除主结论
- **WHEN** split metadata 无法证明 `input_beam` 与 future label/test label 防泄漏
- **THEN** run metadata 和 summary MUST 标记该 run 不可用于 history-anchored 主结论
- **AND** exclusion reason MUST 包含 `history_window_leakage_unknown` 或等价机器可读原因

### Requirement: History-anchored quick validation protocol
MMW cross-scene adaptation protocol MUST 支持独立的 history-anchored quick validation mode。该 mode MUST 与默认 sensor-assisted quick validation 分开声明、分开汇总，并默认使用一个 source 场景泛化到其它两个 target 场景、两个 seed 和 `label_budget=10` 的最小矩阵。

#### Scenario: 生成 history-anchored quick validation plan
- **WHEN** 用户选择 history-anchored quick validation 配置
- **THEN** plan metadata MUST include `profile=history_anchored_quick_validation` 或等价机器可读标记
- **AND** plan metadata MUST include `budgets=[10]`
- **AND** plan metadata MUST include exactly two seeds unless explicitly overridden
- **AND** plan MUST record source scene、target scenes、history anchor mode 和 residual target mode

#### Scenario: 使用现有 sunny Town10 三场景
- **WHEN** local MMW availability 包含 `Town10_skybridge_seed24`、`Town10_Hroad_seed42` 和 `Town10_crossroad_seed24`
- **THEN** history-anchored quick validation MAY 使用其中一个场景作为 source、其它两个场景作为 target
- **AND** summary MUST 记录这是 scenario-level 泛化，不得声称 leave-one-town-out 或 weather-shift 验证

#### Scenario: quick validation 不自动扩展完整 sweep
- **WHEN** 用户运行默认 history-anchored quick validation 命令
- **THEN** runner MUST NOT 默默扩展到 budgets `[0,5,10,20,50]` 或三个以上 seeds
- **AND** 若用户通过 CLI override 扩展矩阵，plan metadata MUST 记录 override 来源

### Requirement: History-anchored adaptation 防泄漏
history-anchored target adaptation MUST 遵守 MMW target adaptation protocol。`label_budget=0` 不得读取 target future beam 作为 supervised residual loss；`label_budget>0` 只能从 target_adapt labeled subset 读取 future beam label；target_test MUST 始终只用于最终评估。

#### Scenario: 0-label residual adaptation 禁止 target future label
- **WHEN** history-anchored target adaptation 的 `label_budget=0`
- **THEN** adaptation MUST NOT 读取 target_adapt future beam label 计算 residual supervised loss
- **AND** adaptation MAY 使用 entropy、consistency、prototype confidence 或其它不读取 target future label 的无监督 loss
- **AND** metadata MUST 记录 `used_target_beam_for_supervised_loss=false`

#### Scenario: few-shot residual adaptation 只读 labeled target_adapt
- **WHEN** history-anchored target adaptation 的 `label_budget>0`
- **THEN** supervised residual loss MUST 只使用 sampled labeled target_adapt subset
- **AND** sampling manifest MUST 记录 sample id、last_beam、future beam、residual label、coarse sector、relative azimuth bin 和 sampling seed
- **AND** 未选中的 target_adapt 样本 MAY 作为 unlabeled adaptation 数据但 MUST NOT 读取其 future beam label 作为监督

#### Scenario: target_test 只参与最终评价
- **WHEN** history-anchored source-only 或 adapted target_test evaluation 执行
- **THEN** target_test future beam label MAY 用于计算最终 metrics
- **AND** target_test future beam label、beam_power、path fields 和 radio labels MUST NOT 用于 adaptation threshold selection、prototype update、temperature fitting 或 early stopping

### Requirement: History-anchored run metadata
MMW LOSO run artifacts MUST 记录 history-anchored residual 的关键配置和 eligibility 字段，使后续分析可以复现、过滤和诊断 absolute-ID prior collapse。

#### Scenario: run metadata 记录 history 配置
- **WHEN** history-anchored run 完成任一 stage
- **THEN** run metadata MUST 包含 `history_anchor_enabled`、`history_anchor_mode`、`residual_target_enabled`、`num_delta_classes`、`uses_input_beam_as_model_input` 和 enabled sensing modalities
- **AND** metadata MUST 包含 source scenes、target scene、budget、seed、seq_len、num_pred 和 split eligibility

#### Scenario: summary 记录 collapse 诊断输入
- **WHEN** history-anchored summary 汇总 source-only absolute baseline 和 residual run
- **THEN** summary MUST include source train beam histogram、target test beam histogram 和 model predicted beam histogram 或对应 artifact path
- **AND** summary MUST 能据此标记 source prior collapse、history residual recovery 或不可判定原因

#### Scenario: artifacts 保留配置快照
- **WHEN** history-anchored run 启动
- **THEN** run output directory MUST 保存 resolved config 或等价配置快照
- **AND** 配置快照 MUST 包含 history anchor、residual loss、private calibration 和 dataloader split 参数
