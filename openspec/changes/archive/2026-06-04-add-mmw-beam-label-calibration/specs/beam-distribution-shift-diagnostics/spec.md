## ADDED Requirements

### Requirement: Beam distribution diagnostics declare label space
分布诊断、GPS-angle correspondence 图和 prediction error label distribution MUST declare whether histograms and distances use raw or calibrated beam label space. 启用 MMW calibration 时，系统 MUST 支持按 calibrated label 计算 histogram 和 ordered/circular distance。

#### Scenario: calibrated histogram 输出
- **WHEN** distribution shift analysis receives an enabled MMW beam calibration config
- **THEN** output histograms MUST include calibrated absolute beam labels
- **AND** metrics JSON MUST record mapping parameters、mapping fingerprint and label space name

#### Scenario: raw 与 calibrated histogram 不混合
- **WHEN** analysis output contains both raw and calibrated histograms
- **THEN** each histogram MUST have an explicit label_space field
- **AND** summary MUST NOT compare raw-source histogram with calibrated-target histogram as a single distance metric

#### Scenario: GPS-angle 图声明 beam index mode
- **WHEN** GPS-angle correspondence visualization plots beam labels
- **THEN** figure summary MUST record `beam_index_mode` as raw or calibrated
- **AND** calibrated plots MUST record direction、offset、num_classes and fit source

#### Scenario: prediction error label distribution 使用一致 label space
- **WHEN** prediction error label distribution is generated from calibrated prediction artifacts
- **THEN** true labels and predicted labels MUST be interpreted in the same calibrated label space
- **AND** tolerance-based adjacent correctness MUST use the calibrated class topology
