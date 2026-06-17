## Context

项目已经形成了一套强治理结构：`AGENTS.md` 规定操作规则，`docs/agent_navigation.md` 提供 AI/maintainer 导航，`docs/project_surface_inventory.md` 维护表面积和 lifecycle inventory，OpenSpec specs 记录需求契约，`tests/test_architecture_boundaries.py` 通过静态检查防止旧入口和退役路线回流。

当前痛点不是缺少规则，而是规则的结构化程度不够。许多事实同时出现在 Markdown 表格、OpenSpec requirements 和 pytest 常量中，例如脚本入口 allowlist、root fusion config allowlist、模型注册例外、batch/runtime 分支、热点预算和 capability lifecycle。AI agent 每次接手非平凡改动时，需要从多处长文档和测试代码中重建项目状态，容易遗漏“current/supporting/retired”边界或把局部打开文件当成权威入口。

本 change 引入中心化 maintainer context index：一份可读、可 diff、可测试消费的结构化文件，作为非平凡改动前的快速入口和健康检查数据源。它不替代 OpenSpec requirements，也不替代 inventory 的审计说明；它只集中保存“需要被机器和 agent 稳定读取”的项目治理事实。

## Goals / Non-Goals

**Goals:**

- 提供稳定路径的机器可读 maintainer context index，优先使用 YAML，便于人工审阅和 Python 测试读取。
- 将现有测试中的长期治理 allowlist 逐步迁移到 index，包括 entrypoint lifecycle、fusion root config、model registration exception、batch/runtime extension surface、hotspot budgets 和 health-check commands。
- 更新 `docs/agent_navigation.md`，让 agent 在非平凡改动前先读取 index，再根据任务路由打开 README、inventory、OpenSpec specs 和源码。
- 更新 `docs/project_surface_inventory.md`，保留面向人的审计说明，同时明确哪些清单的机器可读来源迁移到 index。
- 增加轻量 schema/consistency 检查，确保 index 与 OpenSpec specs、AGENTS、inventory、pyproject 和源码文件存在性对齐。
- 保持无运行副作用：不改变训练、评估、预处理、模型 forward、配置解析、数据读取或本地产物清理。

**Non-Goals:**

- 不在本 change 中拆分 `jepa_gps_shortcut_benchmark.py`、`DeepSense6GDataset` 或其它 runtime 热点。
- 不新增或删除公开训练/评估/诊断 CLI。
- 不把 OpenSpec requirements 全量搬进 YAML；行为契约仍以 `openspec/specs/` 为准。
- 不把 README 或 inventory 删除为纯生成文件；它们仍服务人类阅读和审计上下文。
- 不读取真实 `dataset/`、`outputs/`、checkpoint、cache 或 logs 来生成 index。

## Decisions

### Decision 1: 使用 YAML 作为中心化 index，而不是 Python 常量或 Markdown 表格

选择：新增 `docs/maintainer_context_index.yaml`，以普通 YAML 记录治理事实。

理由：

- YAML 对维护者友好，diff 清晰，不需要运行 Python 就能审阅。
- 当前项目已依赖 PyYAML，测试可用 `conda run -n kd_mm_beam pytest ...` 读取。
- 与 Markdown 表格相比，YAML 更适合精确断言、排序检查、字段必填和跨文件一致性校验。
- 与 Python 常量相比，YAML 不会让测试文件继续成为事实源，也减少 Codex 修改测试逻辑时误改治理数据的风险。

备选：

- Markdown frontmatter 或 Markdown 表格：保留给人读很好，但机器解析脆弱。
- JSON：机器友好但人工编辑体验差，注释能力弱。
- Python 模块：容易被误认为运行时 API，也会把治理数据继续藏在代码里。

### Decision 2: index 是“路由和治理事实索引”，不是行为规范全文

选择：index 记录路径、分类、lifecycle、验证命令、允许集合和热点预算；具体行为契约仍链接到 OpenSpec spec、README 或 inventory 段落。

理由：

- 防止新文件膨胀成第二份 OpenSpec。
- 让 AI agent 快速定位“应该读什么”和“什么不能做”，但仍回到 OpenSpec 看 requirement。
- 避免同一 requirement 在 spec 和 YAML 中双写后出现语义漂移。

备选：

- 把所有 capability requirements 摘要进 index：短期方便，长期会制造新的事实漂移。
- 只写导航文档不新增 index：保持现状，无法解决测试 allowlist 和文档表格重复维护。

### Decision 3: 架构边界测试读取 index，但仍保留行为断言逻辑

选择：把 `tests/test_architecture_boundaries.py` 中的长期治理数据迁移为从 YAML 加载；测试继续实现存在性、非法回流、文档 drift 和轻量导入边界断言。

理由：

- 数据和断言分离后，新增入口或模型例外时 diff 更清楚。
- 测试仍是防线，不会变成简单“YAML 存在即可通过”。
- Codex 能从 YAML 直接知道新增某类对象需要同步哪些字段。

备选：

- 完全生成架构边界测试：过度工程化，不利于审阅。
- 继续在测试顶部维护常量：成本低，但没有解决重复事实源问题。

### Decision 4: index 与 inventory 双向引用但职责不同

选择：`docs/project_surface_inventory.md` 继续记录解释性基线、热点说明和历史上下文；index 提供可测试消费的表。inventory 必须指向 index，index 也必须声明哪些字段由 inventory 或 specs 解释。

理由：

- inventory 有大量自然语言 caveat，不适合全部结构化。
- index 需要简洁稳定，不能承载所有历史背景。
- 双向引用能让人类和 agent 都知道“先看 index 定位，再看 inventory 理解原因”。

备选：

- 让 inventory 成为唯一数据源：Markdown 解析脆弱。
- 让 index 取代 inventory：会丢失审计叙述和暂缓原因。

### Decision 5: 首批迁移选择低风险治理表

选择：首批迁移以下字段：

- `entrypoints.python_allowlist`
- `entrypoints.shell_allowlist`
- `configs.fusion_root_allowlist`
- `models.registration_allowlist`
- `batch_runtime.function_allowlist`
- `hotspots.symbol_budgets`
- `hotspots.file_budgets`
- `health_checks.quick_commands`
- `routing.task_types`
- `retired_routes.tokens`

理由：

- 这些字段当前已经在测试或 inventory 中以列表形式存在，迁移风险低。
- 它们是 Codex 最常需要判断的“能不能新增/要改哪里”的上下文。
- 不涉及 runtime 行为，不会改变训练结果。

后续 change 可以继续迁移更复杂的文档 caveat 或 result-claim governance，但本 change 不做。

## Risks / Trade-offs

- [Risk] 新增 index 后变成第三套事实源。  
  → Mitigation: spec 要求 index 只保存可测试治理事实；OpenSpec requirements 仍是行为权威，inventory 仍是审计解释。测试验证 inventory/AGENTS 指向 index，并检查 lifecycle 覆盖。

- [Risk] YAML schema 过重，维护成本超过收益。  
  → Mitigation: 首批只用轻量必填字段和存在性检查，不引入外部 schema 依赖；后续按需要再增加严格校验。

- [Risk] 迁移测试常量时误放宽架构边界。  
  → Mitigation: 先保持断言语义不变，只替换数据来源；运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 对比。

- [Risk] index 内容太长，Codex 仍然读不动。  
  → Mitigation: 按 section 分组，路由表在前，详细 allowlist 在后；导航文档指导按任务读取相关 section。

- [Risk] active change 与 index 更新顺序冲突。  
  → Mitigation: tasks 要求先在 change artifact 说明新增入口/例外，再更新 index，最后更新测试和 inventory。

## Migration Plan

1. 新增 `docs/maintainer_context_index.yaml`，填入首批治理数据和任务路由。
2. 新增轻量读取/校验 helper，优先放在测试侧或 `tests/` 内，避免成为 runtime API。
3. 更新架构边界测试，使现有 allowlist 和预算从 index 读取。
4. 更新 `docs/agent_navigation.md`，把 index 放入非平凡改动前的检查顺序。
5. 更新 `docs/project_surface_inventory.md`，说明机器可读治理表迁移到 index，inventory 保留审计解释。
6. 运行 OpenSpec validate 和架构边界 focused tests。

Rollback 很简单：如果迁移中发现 schema 不稳定，可保留新增 index 文档但暂时让测试继续读取原常量；行为代码没有改动，不影响 runtime。

## Open Questions

- 首批实现是否将 YAML reader 放在 `tests/` 私有 helper，还是放在 `src/kd_sensing/utils/`？倾向放在 `tests/`，避免把治理文件变成运行时 API。
- 是否在本 change 中迁移全部 `tests/test_architecture_boundaries.py` 常量，还是先迁移 entrypoint/config/model/hotspot 四类？倾向首批迁移低风险核心集合，避免一次性大改测试。
