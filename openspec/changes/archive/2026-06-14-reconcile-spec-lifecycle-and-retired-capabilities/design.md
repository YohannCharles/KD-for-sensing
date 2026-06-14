## Context

项目已经有较强的健康护栏：OpenSpec strict validate、架构边界测试、入口 allowlist、退役路线拒绝和本地产物边界都能运行通过。但当前 `openspec/specs/` 同时保存当前能力、支撑能力和退役墓碑 capability，且部分历史 requirements 仍残留 active wording，例如把 HiST-Beam LOSO 或 Raymobtime 写成当前推荐入口或 active mainline。

这类问题不是语法错误，也不一定会导致测试失败，却会显著增加 AI agent 和维护者的理解歧义。尤其当 `openspec list --json` 显示无 active change，而工作树仍包含未跟踪 archived change、新 spec、未提交源码或 ignored `__pycache__` 时，agent 容易把“历史记录”“本地噪声”和“当前契约”混在一起。

## Goals / Non-Goals

**Goals:**

- 建立 OpenSpec capability 生命周期分类，明确 current、supporting、retired-tombstone 的读取语义。
- 用中心化 lifecycle inventory 覆盖当前 `openspec/specs/`，避免每次读到退役 capability 文件名时误判为当前入口。
- 清理 `project-architecture` 中与 README/inventory/current specs 冲突的旧 active wording。
- 在 `docs/agent_navigation.md` 中加入 lifecycle-first 读取规则、墓碑 spec 识别规则和本地状态噪声判断规则。
- 扩展架构边界测试，使退役墓碑 spec 的 active wording 回流、未分类 spec 和 lifecycle inventory 漂移能被快速发现。

**Non-Goals:**

- 不恢复 HiST/Hist、Raymobtime s008、Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D、Multimodal-NF 或旧 KD 入口。
- 不删除历史 archive、历史研究笔记、旧本地 outputs/logs/checkpoint 或 ignored cache。
- 不重构训练、评估、数据集、模型、配置加载或运行产物 schema。
- 不要求本 change 完成所有热点 runtime 模块拆分；只记录与 lifecycle/歧义相关的热点和后续路线。

## Decisions

### 1. 使用中心化 lifecycle inventory，而不是给每个 spec 增加强制头部字段

实现时在 `docs/project_surface_inventory.md` 或等价文档中新增 OpenSpec capability lifecycle 分类，覆盖每个 `openspec/specs/<capability>/spec.md`。分类至少包含：

- `current`: 当前可运行、可推荐或当前需求契约能力。
- `supporting`: 不作为独立入口推荐，但为当前 workflow 提供支撑代码、数据契约、metric 或 migration guard。
- `retired-tombstone`: 只保留退役、拒绝、迁移边界和防回流说明，不是当前运行能力。

理由：逐个改 76 个 spec 文件会造成大量机械 churn；中心化 inventory 更符合现有项目表面积审计模式，也便于架构测试做完备性检查。替代方案是在每个 spec 顶部增加 `Lifecycle:` 字段，语义最局部但改动面过大，后续可以在 inventory 稳定后再迁移。

### 2. 退役墓碑 spec 允许存在，但必须“一眼可辨”

`retired-tombstone` spec 的 Purpose 或首个 requirement MUST 明确写出已退役/不属于当前支持能力。它可以记录旧名称、旧配置、旧 CLI、migration guard、支撑代码保留原因和禁止回流规则，但不得使用未加退役限定的“当前推荐”“active mainline”“默认入口”“可运行 workflow”等措辞。

理由：保留墓碑 spec 有价值，因为它能防止旧研究线回流；问题不在“存在”，而在“看起来像当前能力”。替代方案是删除退役 spec，但会丢失迁移和拒绝契约，也会削弱防回流测试。

### 3. supporting capability 明确不等于当前入口

部分能力介于 current 和 retired 之间，例如 BGAM 依赖的 TopK candidate manifest/loss 支撑代码、通用 circular metrics、migration guard 或 cleanup manifest。它们可以保留实现和测试，但文档 MUST 说明不是 standalone 推荐入口，不能通过 console script、root config 或 README quickstart 重新暴露。

理由：这解决“Top8 selector 退役，但 TopK 支撑代码仍在”的中间态。替代方案是二分成 current/retired，会把必要支撑代码也误删或误归入当前入口。

### 4. `project-architecture` 只描述当前架构和退役边界

实现时要修改旧 requirements：凡是把 HiST/Hist、Raymobtime s008、Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D 或 Multimodal-NF 描述成 active mainline、当前推荐入口、长期 orchestration 或当前热点的内容，必须改为退役禁止回流、supporting-only 或历史背景。当前主线描述应与 README 和 inventory 对齐到 Image+GPS JEPA query-pool、paired baseline/control、Vision-Position baseline suite、BGAM、MMW GPS v2、CSI hardening、viewer manifest、通用训练评估和明确保留的支撑能力。

理由：`project-architecture` 是高优先级当前 spec，一旦里面残留旧 active wording，AI 会合理地信它。替代方案是在导航文档里提醒忽略旧段落，但那会制造“spec 自己不可信”的更大问题。

### 5. 健康护栏检查语义漂移，而不检查本地未跟踪产物本身

架构测试应检查已跟踪文档和 specs 的 lifecycle inventory 完整性、退役墓碑 wording、当前 spec 中旧 active wording，以及 README/inventory/navigation 的引用一致性。它不应因为本地存在 ignored `__pycache__`、`.pytest_cache` 或未跟踪实验输出而失败；这些属于本地状态噪声，应通过导航文档说明如何识别。

理由：CI 和日常测试应稳定、无副作用；本地缓存清理不应成为源码契约。替代方案是测试扫描 `git status --ignored`，容易因每个开发者本地状态不同而抖动。

## Risks / Trade-offs

- [Risk] lifecycle inventory 又变成一份需要维护的目录清单。→ Mitigation：只维护 capability lifecycle，不重复源码目录；架构测试检查每个 spec 文件都被分类，减少漂移。
- [Risk] retired-tombstone 与 supporting 的边界可能争议。→ Mitigation：以是否能作为 standalone 当前入口推荐为分界；支撑代码只能被当前 workflow 消费，不得拥有旧 CLI/root config。
- [Risk] 清理 `project-architecture` 旧段落时误删仍有效的通用 LOSO/few-shot 或 MMW 支撑语义。→ Mitigation：把“Hist 专用”与“通用 cross-scene/few-shot”分开；如仍保留 `engine/loso_data.py`，应记录为 supporting 或待决，不把 Hist workflow 复活。
- [Risk] 退役 spec 文件名仍会诱发误读。→ Mitigation：inventory、导航文档和墓碑 spec 首段都明确 lifecycle；必要时后续再评估是否重命名或合并墓碑 spec。

## Migration Plan

1. 在 OpenSpec delta 中定义 lifecycle 分类、退役墓碑 wording、supporting capability 边界和健康护栏要求。
2. 更新 `docs/project_surface_inventory.md`，新增 OpenSpec capability lifecycle 分类，覆盖所有 current specs。
3. 清理 `openspec/specs/project-architecture/spec.md` 中旧 active wording，并保留退役禁止回流要求。
4. 更新 `docs/agent_navigation.md` 和 `openspec/specs/ai-maintainer-navigation/spec.md`，明确 lifecycle-first 读取流程。
5. 扩展 `tests/test_architecture_boundaries.py`，验证 lifecycle inventory 完整性、墓碑 wording、当前文档不把退役路线描述为推荐入口。
6. 运行 `openspec validate reconcile-spec-lifecycle-and-retired-capabilities --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，必要时补跑文档/配置相关 focused tests。

Rollback 策略：本 change 只修改文档、OpenSpec 和静态测试；若检查规则过严导致误报，可回退新增检查或将争议 spec 临时标记为 `supporting` 并在 inventory 记录原因，不需要回滚 runtime。

## Open Questions

- 是否在本 change 中把 lifecycle inventory 放入现有 `docs/project_surface_inventory.md`，还是新增独立 `docs/openspec_lifecycle_inventory.md` 后再由 inventory 引用？
- 对于 `deepsense6g-gps-top8-candidate-selector` 这类“入口退役但支撑代码保留”的 spec，最终分类使用 `supporting` 还是拆成 retired tombstone + supporting spec？
- 是否要把已完成但未提交的 `refactor-modality-difficulty-pipelines` 归档状态作为本 change 的验证案例，还是只在 navigation 中写成通用状态判断规则？
