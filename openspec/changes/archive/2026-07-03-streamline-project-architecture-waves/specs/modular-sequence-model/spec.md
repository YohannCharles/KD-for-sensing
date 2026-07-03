## ADDED Requirements

### Requirement: ModularSequenceModel forward is staged internally
`ModularSequenceModel.forward` MUST 保持当前 public signature 和 output contract，但内部 MUST 拆分为可测试、职责明确的 stages：raw/reliability input collection、encoder dependency resolution、encoder/projector execution、representation core input assembly、head execution、geometry prior/rerank post-processing、diagnostics/runtime metadata assembly 和 auxiliary outputs。

#### Scenario: Forward 输出兼容
- **WHEN** staged forward 处理当前单模态、fusion、token-aware、geometry-prior、safe-rerank、AMBER full、predictive GPS-query 或 missing-modality 配置
- **THEN** `logits`、`input_features`、`output_features`、`modalities`、`modality_features`、`encoder_features` 和已登记 diagnostics keys MUST 保持兼容
- **AND** `adapt_model_output` MUST 不需要新增 model-specific 分支才能消费 staged forward 输出

#### Scenario: Stage helper 不变成 public API
- **WHEN** forward stage helper 被提取
- **THEN** helper MUST 保持在 modular model owner 或职责明确的内部模块中
- **AND** README、docs 和 tests MUST 不把 stage helper 描述为外部 public API

### Requirement: New components do not expand main forward routing
新增 encoder、projector、representation core、head、geometry prior helper 或 diagnostics metadata MUST 优先通过组件 metadata、capability flags 和 existing stage hooks 接入。普通 component baseline MUST 不向主 forward 添加 baseline-specific 参数、condition id 分支或 private output assembly 分支。

#### Scenario: 新 encoder 使用 declared dependencies
- **WHEN** 新 encoder 需要上下文模态、reliability metadata、visual token diagnostics 或 temporal auxiliary metadata
- **THEN** encoder MUST 通过声明式 dependency/capability metadata 或现有 reliability hook 暴露需求
- **AND** main forward MUST 不新增只服务该 encoder 的硬编码分支

#### Scenario: 新 core 使用 staged assembly
- **WHEN** 新 representation core 需要 spatial modality tokens、missing modality metadata 或 token readout diagnostics
- **THEN** core MUST 通过 capability flag 和 staged core-input assembly 接入
- **AND** 普通 baseline 未启用该 core 时 MUST 不需要提供新增 metadata

### Requirement: Forward metadata remains auditable
staged forward MUST 保持 training strategy metadata、runtime metadata、diagnostics payload 和 architecture summary 可审计。新增 metadata 字段 MUST 标明生产组件、消费组件和是否影响 comparability。

#### Scenario: Metadata 来源可追踪
- **WHEN** forward 输出 encoder runtime metadata、geometry prior diagnostics、rerank diagnostics、feature consistency diagnostics、token readout diagnostics 或 AMBER auxiliary payload
- **THEN** payload MUST 能追踪到对应 encoder/core/post-processing owner
- **AND** focused tests MUST 覆盖至少一个 metadata-producing model path

