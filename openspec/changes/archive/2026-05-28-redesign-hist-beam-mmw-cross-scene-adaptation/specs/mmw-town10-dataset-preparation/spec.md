## ADDED Requirements

### Requirement: MMW local download processing
MMW Town10 preparation MUST 通过显式配置或 CLI overrides 支持位于 `dataset/_downloads/MMW/<condition>/Sensor_Data` 和 `dataset/_downloads/MMW/<condition>/Channel_Data` 的 zip inputs。Processing MUST 将 prepared artifacts 写入 `dataset/MMW/<condition>/Prepared/<scenario>`，并 MUST 不移动或删除已下载 zip 文件。

#### Scenario: 处理 sunny 已下载 zip
- **WHEN** 用户提供 sensor zip `dataset/_downloads/MMW/sunny/Sensor_Data/Town10_skybridge_seed24.zip` 和 channel zip `dataset/_downloads/MMW/sunny/Channel_Data/Town10.zip`
- **THEN** preparation MUST extract or index them into `dataset/MMW/sunny/Sensor_Data` and `dataset/MMW/sunny/Channel_Data`
- **AND** prepared artifacts MUST be written under `dataset/MMW/sunny/Prepared/Town10_skybridge_seed24`
- **AND** metadata MUST record source zip absolute paths and fingerprints

### Requirement: Sensor/channel scenario alias matching
MMW preparation MUST explicitly handle cases where sensor scenario names and channel scenario directories differ, such as sensor `Town10_skybridge_seed24` and channel `Town10/Town10_skybridge`. Matching MUST be based on declared alias, frame id and CAV agent, and MUST be recorded in metadata.

#### Scenario: channel agent 匹配正确
- **WHEN** a sensor frame belongs to agent `cav_1` and frame `008362`
- **THEN** preparation MUST prefer channel paths under channel agent `cav_1` for frame `008362`
- **AND** it MUST NOT silently match the frame to `cav_2` or `cav_3`
- **AND** frame manifest MUST make the matched channel agent auditable

#### Scenario: alias 匹配写入 metadata
- **WHEN** sensor scenario and channel scenario names differ but are matched by alias
- **THEN** metadata MUST record sensor scenario, channel scenario, alias rule and matched frame count
- **AND** unmatched frames MUST be counted by reason

### Requirement: Prepared artifact validity checks
MMW preparation MUST run validity checks before declaring prepared status. Checks MUST include finite beam power vectors, non-empty sequence windows, CAV/channel agent consistency, frame continuity and required modality coverage.

#### Scenario: agent 错配导致失败
- **WHEN** frame manifest contains a CAV agent whose channel path points to a different CAV agent without explicit override
- **THEN** preparation MUST fail or mark the artifact invalid
- **AND** sanity report MUST include examples of mismatched rows

#### Scenario: 有效 prepared summary
- **WHEN** preparation succeeds
- **THEN** sanity report MUST include valid frame count, window count, agent frame counts, modality coverage, channel failure counts, beam histogram, train/test window counts and artifact paths

### Requirement: Incremental MMW preparation
MMW preparation MUST support incremental processing as additional condition/town/scenario zips arrive. Incremental processing MUST preserve existing prepared artifacts unless `force` is explicitly requested for the same condition/scenario.

#### Scenario: 新 condition 到达
- **WHEN** rainy or foggy zip files become available after sunny has already been prepared
- **THEN** preparation MUST process the new condition into its own `dataset/MMW/<condition>` directory
- **AND** existing sunny prepared artifacts MUST remain unchanged unless explicitly forced

#### Scenario: force 重建单个 scenario
- **WHEN** user requests force rebuild for `sunny/Town10_skybridge_seed24`
- **THEN** preparation MAY overwrite that prepared scenario artifacts
- **AND** it MUST not remove other condition or scenario prepared artifacts
