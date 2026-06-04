## ADDED Requirements

### Requirement: 历史 GPS coarse/pseudo label 序列导出
GPS coarse anchor workflow MUST 支持将单帧 GPS coarse anchor 扩展为历史窗口 pseudo label 序列，供 BGAM、residual/fusion 和 diagnostics 消费。该导出 MUST 保持现有 GPS anchor anti-leakage 边界，只使用 GPS/pose、RSU pose、历史时间戳、frozen GPS prior 和合法 calibration metadata。

#### Scenario: 导出历史 coarse anchor artifact
- **WHEN** 用户启用 `export_history_pseudo_labels`
- **THEN** 系统 MUST 写出 history pseudo label artifact 或 manifest 字段
- **AND** 每个样本 MUST 包含 `history_pseudo_beams` 或 `history_coarse_groups`
- **AND** 每个样本 MUST 包含 `history_pseudo_confidence`、`history_pseudo_entropy`、`history_valid_mask` 和 `history_timestamps`
- **AND** metadata MUST 记录 history length、prediction horizon、pseudo label source、label-space 和 mapping fingerprint

#### Scenario: downstream 可消费 anchor 序列
- **WHEN** downstream BGAM 或 residual/fusion 模型启用 GPS pseudo-history
- **THEN** batch 或 model input MUST 能提供历史 pseudo label 序列、置信度和 valid mask
- **AND** 缺失历史 pseudo label 时系统 MUST 抛出清晰错误或记录配置允许的 fallback

#### Scenario: query label 不参与历史导出
- **WHEN** 系统为 target query/test 样本导出历史 GPS pseudo label
- **THEN** target query/test 真实 beam label MUST NOT 参与 pseudo label 生成、校准、normalization fit 或参数选择
- **AND** metadata MUST 记录 `query_label_used_for_pseudo_history=false`
