## ADDED Requirements

### Requirement: Radio prototype 不依赖旧解耦 baseline
Radio-semantic adaptation MUST 保留为 V6 radio prototype baseline 或 fallback，但 MUST NOT 依赖旧 `v3_decoupled` source-only baseline、旧 shared/private scene classifier、orthogonality loss、shared scene confusion loss 或 private scene preservation loss。radio prototype 的有效性 MUST 由 radio label/prototype coverage、radio loss、beam metrics 和防泄漏 metadata 诊断。

#### Scenario: radio adaptation source mapping 不返回 v3
- **WHEN** LOSO runner 为 `v6_radio_proto` 或 `adapter_radio_proto` 选择 source checkpoint
- **THEN** source checkpoint MUST 来自合法非旧解耦 variant 或用户显式指定的合法 source variant
- **AND** 系统 MUST NOT 自动选择 `v3_decoupled`

#### Scenario: radio loss 不计算旧 scene loss
- **WHEN** source radio training batch 同时包含 scene label 和 radio_semantic_label
- **THEN** 系统 MAY 计算 radio semantic CE
- **AND** 系统 MUST NOT 因 scene label 存在而计算旧 shared scene confusion 或 private scene preservation loss
