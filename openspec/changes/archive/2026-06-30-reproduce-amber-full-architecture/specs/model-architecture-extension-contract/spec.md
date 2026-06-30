## ADDED Requirements

### Requirement: AMBER full 默认使用 component baseline 路径
AMBER full architecture reproduction MUST 默认通过 `modular_sequence` 及其 encoder/projector/representation core/head/loss 组件实现。只有当 active design 证明该架构无法通过组件边界表达时，系统 MAY 使用 whole-model exception；该例外 MUST 提供额外 registry、forward、metadata、ModelOutput adaptation 和架构摘要测试。

#### Scenario: AMBER full 作为 representation core 构建
- **WHEN** AMBER full 变化集中在 fusion transformer、mask attention、CMA payload 或 beam head 输入表示
- **THEN** 实现 MUST 优先新增或扩展 `REPRESENTATION_CORES`、loss/objective helper 和配置
- **AND** 系统 MUST 不复制 dataset 解析、训练循环或专用 batch forward 分支

#### Scenario: whole-model exception 需要设计理由
- **WHEN** 实现者决定为 AMBER full 新增完整 `MODELS.register(...)` 名称
- **THEN** OpenSpec design 或后续 artifact MUST 说明不能使用 component baseline 的具体原因
- **AND** tasks MUST 包含 registry build、synthetic forward、`adapt_model_output`、metadata、architecture summary 和 architecture boundary tests
