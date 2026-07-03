## ADDED Requirements

### Requirement: New model extensions must avoid shared hotspot edits by default
新增普通 baseline、component baseline 或 workflow baseline MUST 默认不修改 dataset 主体、training loop、evaluation loop、batch runtime 主路由或 `ModularSequenceModel.forward` 主体。确需修改共享热点时，OpenSpec design/tasks MUST 说明原因、影响面、focused tests 和 public behavior compatibility。

#### Scenario: 普通 component baseline
- **WHEN** baseline 只替换 encoder、projector、core、head、loss、metadata 或 config recipe
- **THEN** 实现 MUST 限定在对应组件 owner、registry、config/spec/test
- **AND** 不得修改 dataset class 主体、trainer 主循环或 evaluation loop

#### Scenario: 共享契约确需扩展
- **WHEN** 新能力确实需要新增 batch field、model forward metadata 或 evaluation schema 字段
- **THEN** change artifact MUST 同步更新 modality/batch/runtime/model extension specs
- **AND** focused tests MUST 覆盖普通 baseline 忽略新增字段和 opt-in baseline 消费新增字段两种路径

### Requirement: Model architecture summary covers refactored components
重构或新增模型组件后，模型架构摘要 MUST 继续能审计 registry id、组件 role、参数量、trainable params、checkpoint/freeze policy、reliability metadata consumption 和 comparability metadata 来源。内部模块移动 MUST 不让 summary 回落为 unknown，除非 design 明确说明无法自动分组。

#### Scenario: 组件移动后摘要稳定
- **WHEN** encoder/core/head 或 whole-model exception owner 文件被移动、拆分或合并
- **THEN** architecture summary focused tests MUST 继续验证对应 registry type、class path、role 和参数量字段
- **AND** docs/model architecture inventory MUST 与 current registry surface 保持一致

### Requirement: Whole-model exceptions remain explicit after cleanup
删除 facade、合并 helper 或阶段化 forward 后，仍保留的 whole-model exception MUST 继续有 current spec、active design、inventory 或 focused test 说明。退役整模型 direct import、alias 或 removed wrapper 不得作为包结构保留对象。

#### Scenario: Whole-model exception audit
- **WHEN** cleanup 后扫描 `@MODELS.register(...)`
- **THEN** 每个完整模型注册名 MUST 能映射到 current capability、explicit exception 或 workflow/paper reproduction 边界
- **AND** 无 current 依据的旧整模型 class MUST 删除或从 registry surface 退出

