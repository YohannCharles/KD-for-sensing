## ADDED Requirements

### Requirement: MMW radio-semantic derivation metadata
MMW preparation and dataset runtime MUST expose enough metadata to derive radio-semantic labels from channel-derived beam power vectors. The system MUST record whether each frame or sequence can derive a radio label, the builder mode/config version, and the unavailable reason when derivation fails.

#### Scenario: frame manifest supports radio label derivation
- **WHEN** frame manifest contains a finite 64-beam power vector path and beam label for a CAV frame
- **THEN** manifest or derived metadata MUST identify that the frame is eligible for radio-semantic label construction
- **AND** metadata MUST include `num_beams`, codebook/profile information, beam_power path and label source

#### Scenario: derivation unavailable is explicit
- **WHEN** channel file cannot produce finite beam power or the beam power file is missing
- **THEN** preparation or dataset metadata MUST mark radio-semantic derivation as unavailable
- **AND** metadata MUST record a machine-readable reason such as missing beam power, invalid power vector or unsupported channel fields

### Requirement: MMW radio-semantic labels are not sensing inputs
MMW dataset configuration MUST keep radio-semantic labels, CSI/channel paths and beam_power separate from sensing input modalities. Enabling radio-semantic training MUST NOT implicitly enable CSI/channel/beam_power as model input.

#### Scenario: radio label enabled without channel input
- **WHEN** user enables `radio_semantic.enabled: true` for an MMW HiST-Beam run
- **THEN** dataset MAY return `radio_semantic_label` and `beam_power` for labels or metrics
- **AND** model input modalities MUST remain limited to the configured sensing modalities

#### Scenario: channel-derived metrics are evaluation-only on target_test
- **WHEN** target_test samples contain beam_power
- **THEN** evaluation MAY compute normalized received power and beam power loss dB
- **AND** target adaptation MUST NOT use target_test beam_power for training, threshold selection or prototype update
