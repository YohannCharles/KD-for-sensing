## ADDED Requirements

### Requirement: MMW preparation records beam label calibration provenance
MMW Town10 preparation MUST preserve raw channel-derived beam label provenance and MAY record calibration candidate metadata without overwriting raw beam power vector semantics.

#### Scenario: frame manifest 保留 raw beam label
- **WHEN** preparation 从 channel 文件生成 beam power vector 和 raw `argmax` label
- **THEN** frame manifest MUST 记录 raw beam label、beam power path、num beams 和 label source
- **AND** preparation MUST NOT rewrite the beam power vector to express calibrated class order

#### Scenario: calibration metadata 可审计
- **WHEN** preparation 或后续诊断产出 scene-level calibration candidate
- **THEN** metadata MUST record direction、offset、num_classes、label_space name、fit source 和算法版本
- **AND** metadata MUST distinguish candidate calibration from the raw label used to generate beam power files

#### Scenario: split metadata 同时说明 raw/calibrated label 分布
- **WHEN** split builder receives an enabled calibration config
- **THEN** split metadata MAY include calibrated label histograms in addition to raw label histograms
- **AND** each histogram MUST declare its label space and mapping fingerprint
