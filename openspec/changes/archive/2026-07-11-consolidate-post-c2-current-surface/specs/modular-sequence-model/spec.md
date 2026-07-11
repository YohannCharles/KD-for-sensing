## MODIFIED Requirements

### Requirement: 模块化模型架构摘要分组
`ModularSequenceModel` MUST 支持 instance architecture summary 识别 current 内部组件。摘要 MUST 按 `encoders.<modality>`、`projectors.<modality>`、`representation_core` 和 `heads.<name>` 分组，并 MUST 保持现有 forward、batch runtime 和 `training_strategy_metadata()` 行为兼容；已退役 geometry prior、logit fusion 和 safe reranker MUST 不再作为可选摘要 role。

#### Scenario: image-only modular summary
- **WHEN** current consumer 对 image-only `modular_sequence` 模型生成架构摘要
- **THEN** 摘要 MUST 包含 image encoder、image projector、representation core 和 beam head 组件
- **AND** 每个组件 MUST 包含 path、class、registry type 或 fallback class name、total params 和 trainable params

#### Scenario: image+GPS modular summary
- **WHEN** current consumer 对 image+GPS `modular_sequence` 模型生成架构摘要
- **THEN** 摘要 MUST 分别包含 image encoder 和 GPS encoder 参数量
- **AND** 摘要 MUST 包含多模态 representation core 参数量

### Requirement: ModularSequenceModel forward is staged internally
`ModularSequenceModel.forward` MUST 保持当前 public signature 和 output contract，但内部 MUST 维持可测试的 raw/reliability input collection、encoder dependency resolution、encoder/projector execution、representation core input assembly、head execution、diagnostics/runtime metadata assembly 和 auxiliary output stages。已退役 geometry/safe-rerank/GPS-query post-processing MUST 从 staged route 删除。

#### Scenario: Forward 输出兼容
- **WHEN** staged forward 处理当前单模态、普通 fusion、token-aware、AMR/AMBER、JEPA mean-context 或 missing-modality 配置
- **THEN** `logits`、`input_features`、`output_features`、`modalities`、`modality_features`、`encoder_features` 和 current diagnostics keys MUST 保持兼容
- **AND** `adapt_model_output` MUST 不需要新增 model-specific 分支

#### Scenario: Stage helper 不变成 public API
- **WHEN** forward stage helper 被保留或收缩
- **THEN** helper MUST 保持在 modular model owner 或职责明确的内部模块中
- **AND** README、docs 和 tests MUST 不把 stage helper 描述为外部 public API

### Requirement: New components do not expand main forward routing
新增 encoder、projector、representation core、head 或 current diagnostics metadata MUST 优先通过组件 metadata、capability flags 和 existing stage hooks 接入。普通 component baseline MUST 不向主 forward 添加 baseline-specific 参数、condition id 分支或 private output assembly 分支。

#### Scenario: 新 encoder 使用 declared dependencies
- **WHEN** 新 encoder 需要上下文模态、reliability metadata、visual token diagnostics 或 temporal auxiliary metadata
- **THEN** encoder MUST 通过声明式 dependency/capability metadata 或现有 reliability hook 暴露需求
- **AND** main forward MUST 不新增只服务该 encoder 的硬编码分支

#### Scenario: 新 core 使用 staged assembly
- **WHEN** 新 representation core 需要 spatial modality tokens、missing modality metadata 或 token readout diagnostics
- **THEN** core MUST 通过 capability flag 和 staged core-input assembly 接入
- **AND** 普通 baseline 未启用该 core 时 MUST 不需要提供新增 metadata

### Requirement: Forward metadata remains auditable
staged forward MUST 保持 current training strategy metadata、runtime metadata、diagnostics payload 和 instance architecture summary 可审计。新增 metadata 字段 MUST 标明生产组件、消费组件和是否影响 comparability；已退役 geometry/rerank/GPS-query payload MUST 不再作为兼容输出义务。

#### Scenario: Metadata 来源可追踪
- **WHEN** forward 输出 encoder runtime metadata、feature consistency diagnostics、token readout diagnostics、missing-mask diagnostics 或 AMBER auxiliary payload
- **THEN** payload MUST 能追踪到对应 encoder/core/head owner
- **AND** focused tests MUST 覆盖至少一个 metadata-producing current model path

### Requirement: ModularSequenceModel forward stage 必须可拆分且行为兼容
ModularSequenceModel 实现 MUST 允许 encoder/projector、core input assembly、core/head execution、current logit handling 和 runtime/auxiliary output assembly 位于窄 helper 中，同时不改变 public forward output。Geometry-prior 与 safe-reranker attachment stage 不再属于 current contract。

#### Scenario: forward 输出兼容
- **WHEN** ModularSequenceModel internals 在删除 retired branch 后运行 current config characterization
- **THEN** logits、current auxiliary outputs、runtime metadata 和 `training_strategy_metadata()` MUST 保持兼容
- **AND** synthetic forward tests MUST 覆盖 ordinary、missing-mask 和 current opt-in metadata 路径
