## ADDED Requirements

### Requirement: MMW codebook topology 必须由可验证的 local-array 证据决定
系统 MUST 为 15 个 MMW domain 审计 `Prepared/<scene>/metadata.json` 的 channel-to-beam metadata、ULA-DFT codebook 参数与 RSU transform。审计 MUST 输出 label-to-local-spatial-frequency/beam-response mapping、相邻性、endpoint 0/63 relation、数据完整性和一个 topology descriptor SHA256。

#### Scenario: 可验证 ULA codebook
- **WHEN** metadata 对所有 domain 一致声明 `ula_dft`、64 beams 和 64 transmit antennas
- **THEN** audit MUST 用显式保存的 DFT convention 构造 local codebook response
- **AND** MUST 输出 index-neighbor 与 endpoint beampattern/local-coordinate relation
- **AND** RSU yaw MUST 只用于 world-to-local GPS 解释，不能改变 beam label mapping

#### Scenario: 元数据或证据不足
- **WHEN** 任一 domain 的 codebook metadata 缺失、不一致或无法重建
- **THEN** audit MUST 标记 physical topology 为 unverified
- **AND** 论文 evidence MUST 不得把 cyclic label index 称为物理角度邻接

### Requirement: topology counterfactual 必须保持除邻接先验外的所有因素相同
系统 MUST 生成 matched BPA variants：linear、cyclic、physical-verified（仅当 audit 通过）和 deterministic random-permuted topology。它们 MUST 共享 head、BPA weight/temperature/sigma、router、seed、split、batch、epoch、training missing protocol 和 fixed eval cache。

#### Scenario: topology paired ablation
- **WHEN** launcher 构建 topology ablation config
- **THEN** config MUST 记录 topology descriptor、mapping checksum、matched control 和 allowlisted diff
- **AND** evaluator/summary MUST 拒绝 topology descriptor 或 cache/split identity 不匹配的 paired rows

