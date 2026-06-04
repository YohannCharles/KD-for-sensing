## ADDED Requirements

### Requirement: MMW calibrated hard label loading
MMW dataset MUST support returning calibrated hard beam labels when `data.dataset.beam_label_calibration.enabled=true`. Calibration MUST apply to historical `input_beam` and future `target_beam` while preserving existing tensor shapes and modality-aware loading behavior.

#### Scenario: calibrated input 和 target beam shape 稳定
- **WHEN** MMW dataset 配置 `seq_len=8`、`num_pred=3` 且启用 beam label calibration
- **THEN** 单样本 `input_beam` MUST 仍为长度 8 的整数张量
- **AND** 单样本 `target_beam` MUST 仍为长度 3 的整数张量
- **AND** 所有合法 label MUST 位于 `[0, num_classes)` 的 calibrated label space

#### Scenario: 显式 future_beam_label 字段被映射
- **WHEN** MMW split CSV 包含 `future_beam_label1` 或等价显式 raw label 字段
- **THEN** dataset MUST 在启用 calibration 时将该 raw label 映射为 calibrated `target_beam`
- **AND** metadata MUST preserve the original raw label value for audit

#### Scenario: beam label cache 区分 mapping
- **WHEN** beam label cache 为 eager 或 lazy 且 calibration 配置发生变化
- **THEN** dataset MUST NOT reuse cached calibrated labels from a different mapping fingerprint
- **AND** cache diagnostics MUST record the active mapping fingerprint

#### Scenario: 未启用模态仍不读取
- **WHEN** MMW fusion 配置启用 `["gps", "mmwave"]` 且启用 beam label calibration
- **THEN** dataset MUST only read GPS、mmWave、beam labels and enabled targets
- **AND** calibration MUST NOT cause image、LiDAR、radar、CSI、channel 或 path 文件被额外读取 as sensing inputs
