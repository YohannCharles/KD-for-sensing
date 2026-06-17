## ADDED Requirements

### Requirement: Predictive robustness difficulty preset
系统 MUST 在 shared difficulty pipeline 中提供 Predictive Robustness preset，用于生成 history-aware image/GPS 输入扰动。Preset MUST 支持 canonical `P0-P5` condition，并 MUST 记录足以复现扰动的 metadata。

#### Scenario: 标准化 P-level condition
- **WHEN** profile 或 benchmark suite 引用 `predictive_jepa_robustness` preset
- **THEN** 系统 MUST 标准化 `P0_clean_current`、`P1_current_frame_missing_history_available`、`P2_semantic_occlusion_history_available`、`P3_plausible_wrong_gps_current_image`、`P4_joint_predictive_recovery` 和 `P5_novel_weather_history_available`
- **AND** 每个 condition MUST 映射到现有或新增 image/GPS difficulty operator 参数
- **AND** unknown P-level condition MUST 被拒绝，并在错误中列出可用 condition

#### Scenario: Predictive preset 保持 target 不变
- **WHEN** predictive robustness difficulty profile 应用于 batch
- **THEN** image/GPS tensor、mask、source index、observability score 或 replay metadata MAY 改变
- **AND** `target_beam`、`beam_power`、soft target、sample id 和 split metadata MUST 与输入保持一致

### Requirement: History-aware image missing 和 semantic occlusion
Predictive Robustness image operator MUST 支持当前帧缺失且历史可用、beam-relevant semantic occlusion 和 novel weather/history available condition。Operator MUST 输出 valid mask、observability score、history availability metadata 和 corruption parameters。

#### Scenario: 当前帧缺失但历史可用
- **WHEN** condition 为 `P1_current_frame_missing_history_available`
- **THEN** transform MUST 将当前预测时间步 image 表达为 missing/zero/mask token 或配置声明的 missing expression
- **AND** 历史帧 MUST 保持可用，metadata MUST 记录 history window 和 current frame missing mask

#### Scenario: 语义遮挡可复现
- **WHEN** condition 为 `P2_semantic_occlusion_history_available` 或 `P4_joint_predictive_recovery`
- **THEN** transform MUST 对当前帧应用 deterministic beam-relevant 或 proxy semantic occlusion
- **AND** replay metadata MUST 记录 occlusion ratio、region selection seed、frame range 和是否使用 proxy heuristic

### Requirement: Plausible wrong GPS 扰动
Predictive Robustness GPS operator MUST 支持 plausible wrong GPS，即用同 split 或同场景约束下的邻近但错误 GPS 替换当前 GPS，使其数值看起来可信但指向错误 beam 区域。该扰动 MUST 被标记为 counterfactual input intervention。

#### Scenario: 构造 plausible wrong GPS
- **WHEN** condition 为 `P3_plausible_wrong_gps_current_image` 或 `P4_joint_predictive_recovery`
- **THEN** 系统 MUST 替换或错配 GPS 输入，并记录 source sample、scene constraint、distance/beam offset criteria、seed 和 fallback
- **AND** 替换后的 GPS MUST 不改变当前样本 target label 或 sample id

#### Scenario: 无可用错配样本时降级
- **WHEN** plausible wrong GPS sample pool 不足
- **THEN** 系统 MUST 按配置 skip、fallback 到 deterministic jitter 或拒绝运行
- **AND** warnings MUST 记录 fallback reason 和 affected sample count

### Requirement: Predictive robustness determinism 和 no-future-leak
Predictive Robustness operators MUST 在同 profile、condition、seed、split 和 sample id 下确定性生成扰动，并 MUST 保证 temporal prediction 可用历史不包含未来信息。

#### Scenario: 同 seed 重放一致
- **WHEN** 单元测试对同一 synthetic batch 应用同一 predictive robustness profile 两次
- **THEN** 两次输出的 image/GPS tensors、masks、source indices、observability score 和 metadata MUST 一致

#### Scenario: 历史窗口不使用未来帧
- **WHEN** condition 声明 history window 用于 temporal prediction 或 history availability
- **THEN** 所有 source history index MUST 小于当前预测时间步
- **AND** 若历史不足，metadata MUST 记录不足并按配置 fallback
