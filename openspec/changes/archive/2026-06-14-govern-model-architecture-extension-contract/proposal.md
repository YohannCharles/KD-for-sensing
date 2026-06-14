## Why

当前模型层已经形成 `modular_sequence`、组件 registry、中心化模态契约和共享 forward runtime，但新增 baseline 的默认扩展路径还没有被硬性写成契约。后续 AI agent 或维护者仍可能直接新增整模型、复制 batch/forward 分支或绕过 run metadata，导致模型抽象层和实现层再次漂移。

## What Changes

- 新增模型架构扩展契约，明确新增普通 supervised/adaptation baseline 的默认落点是 `modular_sequence` 及其 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES`、`HEADS` 子组件。
- 将直接新增完整 `MODELS.register(...)` 模型定义为例外路径：必须有 OpenSpec 设计理由、配置入口、forward 输出契约、run metadata 和 focused tests。
- 规定论文复现 / workflow 型 baseline 的边界：可放在 `src/kd_sensing/baselines/<family>/` 或包内 CLI，但不得复制通用训练循环、数据解析或长期脚本入口。
- 将 `training_strategy_metadata()` 或等价 metadata 输出提升为新增可训练模型/组件的审计要求，用于记录 encoder/core/head、冻结策略、checkpoint reuse、reliability metadata 消费和可比性 caveat。
- 收紧扩展文档和 AI 导航：模型/forward/registry 改动必须先看模型架构契约，新增 baseline 指南默认展示模块化配置和组件注册，而不是直接注册整模型。
- 增加健康护栏：架构边界测试应拒绝无例外说明的新整模型注册、未登记的新增模型文件/配置入口、缺少 metadata 的可训练模型、以及新增 batch/forward 特化分支回流。
- 不引入 breaking change；现有 `fusion_strong`、`fusion_lightweight`、`cls_token_transformer_fusion`、`bev_fusion_2604`、JEPA、Vision-Position 和 BeamBench 专用 workflow 继续作为 current/例外或既有能力存在。

## Capabilities

### New Capabilities

- `model-architecture-extension-contract`: 定义新增 baseline、模型组件、整模型例外、论文复现 workflow、metadata 和测试护栏的长期架构契约。

### Modified Capabilities

- `modular-sequence-model`: 增加“新增普通 baseline 默认通过模块化组件扩展”的 requirement，并约束 reliability/adaptive fusion 优先成为可组合组件。
- `component-registry`: 增加新增模型注册治理、整模型例外条件、默认组件导入登记和 registry 发现文档要求。
- `ai-maintainer-navigation`: 增强模型/forward/registry 路由，要求 AI agent 在非平凡模型改动前读取模型架构扩展契约和相关 focused tests。
- `project-health-guardrails`: 增加模型架构护栏检查，防止新整模型、配置入口、batch 分支或文档指南绕开约定。
- `project-architecture`: 明确新增模型不得修改数据解析、训练循环或公共入口，并把论文复现 workflow 与通用可训练 baseline 分层。

## Impact

- OpenSpec：新增 `model-architecture-extension-contract` spec，并为上述 existing capabilities 增加 delta specs。
- 文档：更新 `docs/extension_guide.md`、`docs/agent_navigation.md`、`docs/project_surface_inventory.md`，必要时同步 README 的扩展索引。
- 测试：扩展 `tests/test_architecture_boundaries.py`，新增或扩展模型/registry focused tests，验证新增 baseline 默认走模块化组件、整模型例外可审计、metadata 可用。
- 代码：主要影响 registry/default component 边界、模型 metadata helper、可选的 fusion/adaptive component 注册点和测试工具；不改变现有训练数值语义、dataset split、checkpoint schema 或本地产物边界。
