## Context

项目已经具备较清晰的模型抽象层：`modular_sequence` 负责组合 encoder/projector/core/head，`kd_sensing.registries` 负责轻量组件注册，`kd_sensing.modalities` 维护模态契约，`engine.batch` 与 `engine.runtime` 统一 batch 准备、forward 和输出适配。Scenario D 变更进一步证明 reliability metadata 可以通过 opt-in 方式进入模型，而不需要为每个 difficulty condition 增加专用 forward 分支。

当前风险在扩展治理层：`docs/extension_guide.md` 仍容易让维护者直接新增完整 `MODELS.register(...)` 模型；架构边界测试主要防退役路线和入口漂移，还没有专门检查新增 baseline 是否遵循模块化模型契约。若不收紧，后续新增 baseline 可能重新复制训练循环、数据解析或模型输入分支。

## Goals / Non-Goals

**Goals:**

- 将新增普通 baseline 的默认路径固定为 `modular_sequence` 配置或其子组件注册。
- 保留整模型注册能力，但使其成为可审计例外，而不是默认建议。
- 明确 workflow 型论文复现 baseline 与通用可训练 baseline 的边界。
- 让 AI agent 和维护者在模型/forward/registry 改动前读取同一套架构契约。
- 增加可执行健康护栏，尽早发现新整模型、batch 分支、文档示例或配置入口绕过契约。
- 保持现有 current 模型和 workflow 可用，不改变训练数值语义。

**Non-Goals:**

- 不重写现有 `fusion_strong`、`fusion_lightweight`、`cls_token_transformer_fusion`、JEPA、Vision-Position、BEV-Fusion 或 BeamBench 实现。
- 不删除现有整模型注册名，也不强制所有历史 current 模型立即迁移到 `modular_sequence`。
- 不新增长期训练 CLI、root-level 脚本入口、旧 KD/HiST/residual/Top8 路线或 compatibility facade。
- 不改变 dataset split、target label、checkpoint schema、runtime output layout 或真实 benchmark 产物边界。

## Decisions

1. **把模型扩展治理建成独立 capability，而不是只改文档。**

   新增 `model-architecture-extension-contract` spec，集中定义默认路径、例外路径、metadata、workflow baseline 和测试护栏。这样后续 archive 后仍会成为长期需求契约。备选方案是只更新 `docs/extension_guide.md`，但文档示例不能约束测试和后续 AI 行为。

2. **把新增 baseline 分成四类扩展路径。**

   ```text
   config-only baseline
        └─ 只改 YAML / virtual recipe / hyperparams

   component baseline
        └─ 新增 ENCODERS / PROJECTORS / REPRESENTATION_CORES / HEADS

   whole-model exception
        └─ 新增 MODELS.register，必须有 OpenSpec 理由和 focused tests

   workflow / paper reproduction
        └─ 放在 baselines/<family> 或包内 CLI，不能复制通用训练循环
   ```

   这保留研究灵活性，同时让普通 baseline 不再默认走整模型类。备选方案是禁止新增整模型，但 BEV-Fusion、GPS-conditioned JEPA、BeamBench AE+GPS 这类方法确实有不可压扁成通用 core 的 workflow 或论文结构。

3. **整模型例外由静态护栏和 OpenSpec artifact 管理，不在 runtime 注册表里硬拦截。**

   `MODELS` registry 保持轻量、简单和可测试；架构边界测试扫描新增注册点、公开导出、配置入口和文档 allowlist。例外模型必须在 OpenSpec design 或 spec 中说明为什么不能只新增 encoder/core/head，并提供 forward/output/metadata tests。备选方案是在 registry.register 中加入 runtime policy，但这会把项目治理规则塞进基础注册工具，并影响外部临时实验模块。

4. **新增可训练模型必须输出最小训练策略 metadata。**

   优先复用现有 `training_strategy_metadata()` 模式；`modular_sequence` 自动聚合 encoder/core/head metadata，整模型例外必须提供等价方法。最小 metadata 包含模型类型、启用模态、组件类型、checkpoint reuse/freeze 策略、是否消费 reliability metadata、是否 paper/workflow exception、关键 caveat。这样 run metadata、benchmark comparability 和结果账本可以审计模型差异。

5. **Reliability/adaptive fusion 优先做成可组合组件。**

   Scenario D 的 observability-aware fusion 代表一类后续可能复用的 adaptive fusion 能力。实现阶段优先评估将其接入 `modular_sequence` 的 `REPRESENTATION_CORES`、adapter helper 或新增窄 registry（例如 fusion adapters）。若不新增 registry，也必须在文档和测试中说明它作为显式 opt-in helper 的使用边界。备选方案是让每个 JEPA/CNN 模型手写 gating，但这会破坏可比性和 metadata 统一。

   实现选择：本 change 暂不把 `ObservabilityAwareFusion` 注册为 `REPRESENTATION_CORES`，也不新增 `FUSION_ADAPTERS` registry。它保留为 `kd_sensing.models.observability_aware_fusion` 下的显式 opt-in helper，供 Scenario D benchmark 和后续 reliability-aware 模型在 OpenSpec 说明后调用。原因是当前模块直接操作 image/GPS latent、temporal JEPA fallback 和 condition metadata，尚未满足通用 core 的 `[B, K, T, D]` 输入契约；过早注册会让普通 early-concat、CLS-token transformer、JEPA 和 Vision-Position baseline 的语义边界变模糊。若后续要成为可组合 representation core，应另起 change 定义配置字段、输入 shape、metadata schema、diagnostics 输出和普通 baseline 忽略 reliability metadata 的回归测试。

6. **AI 导航和扩展指南必须从“示例建议”变成“默认路线”。**

   `docs/agent_navigation.md` 的模型路由应指向模型架构契约、`modular_sequence` spec、component registry、batch/runtime 和 focused tests。`docs/extension_guide.md` 的 Add a Model 首例应改为 config-only 或 component-level baseline；直接 `@MODELS.register` 示例只能放在“例外整模型”小节。

7. **健康护栏只检查源码/文档/配置，不读取真实数据或产物。**

   相关测试应基于 tracked Python、YAML、Markdown 和 OpenSpec artifact；不扫描 `dataset/`、`outputs/`、checkpoint 或本地 cache。这样它适合日常快速运行，也符合当前本地产物边界。

## Risks / Trade-offs

- 新护栏可能误伤合理的研究原型。→ 允许 OpenSpec 例外和测试 allowlist，但要求写清楚不能用模块化组件表达的原因。
- `modular_sequence` 可能承载过多职责。→ 新增复杂融合逻辑时优先拆为窄 core/adapter，而不是把所有逻辑堆进 `ModularSequenceModel`。
- Metadata 要求可能增加实现负担。→ 对 `modular_sequence` 自动生成 metadata；整模型例外只要求最小字段。
- 静态扫描可能不完美。→ 先覆盖高置信信号：`@MODELS.register`、新增 `src/kd_sensing/models/*.py`、root fusion YAML、batch/runtime 新分支、extension guide 示例；复杂判断留给 focused tests 和 OpenSpec review。
- 当前已有整模型较多。→ 本 change 不追溯重构它们，只对新增或明显修改的模型路径建立治理。

## Migration Plan

1. 新增并校验 OpenSpec specs，明确 contract、modified capabilities 和 lifecycle 分类。
2. 更新扩展指南和 AI 导航，把默认新增 baseline 路径改为模块化组件。
3. 增加架构边界测试和必要 focused tests，先保护新增路径，不要求迁移所有既有 current 模型。
4. 如实现选择提供 adaptive fusion 可组合入口，补充最小 registry/helper 和 forward tests。
5. 运行 `openspec validate govern-model-architecture-extension-contract --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，并按触碰范围追加模型/registry focused tests。

Rollback 策略：若护栏过严，可保留文档/spec 契约，临时放宽具体静态扫描规则；不得通过恢复旧入口、兼容聚合层或退役研究线来绕过。

## Open Questions

- 整模型例外 allowlist 放在测试常量、OpenSpec spec、inventory，还是一个轻量 manifest 更利于维护？
- `training_strategy_metadata()` 的最小字段是否需要 formal schema helper，还是先用测试校验关键字段存在？
