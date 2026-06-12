## Context

当前项目已经有较完整的说明体系：`README.md` 面向快速上手和主 workflow，`AGENTS.md` 面向 agent 操作规则，`openspec/specs/` 面向需求与架构契约，`docs/project_surface_inventory.md` 面向支持面、热点和生命周期审计。问题不是缺少信息，而是 AI 或新维护者在实际改代码时需要先判断“该信谁、从哪开始、哪些文件不能当作源码权威”。

当前工作树还经常包含 ignored 的本地数据、训练输出、日志、cache、checkpoint、`.egg-info`、`.pytest_cache` 和 active OpenSpec change。对人来说这些上下文可被辨别；对 AI 来说，它们容易造成错误路由，例如从 generated `SOURCES.txt` 推导包结构、从 `outputs/` 反推当前支持入口、或把 `openspec/changes/archive/` 当作当前需求。

## Goals / Non-Goals

**Goals:**

- 提供一份面向 AI agent 和维护者的短导航文档，帮助其在修改代码前快速确定权威来源、当前状态、任务路由、误读边界和验证命令。
- 让导航文档和现有文档分工清晰：AGENTS 继续记录操作规则，README 继续负责 quickstart，OpenSpec specs 继续作为需求权威，inventory 继续作为支持面审计。
- 为导航文档增加轻量架构检查，避免未来删除、漂移或遗漏关键误读防护。
- 明确 generated metadata、ignored runtime artifacts、OpenSpec archive、retired research lines 和 virtual configs 的阅读边界。

**Non-Goals:**

- 不重写 README、AGENTS 或 OpenSpec specs 的整体结构。
- 不新增训练、评估、预处理、诊断或清理 CLI。
- 不修改训练/评估/数据/模型/配置解析 runtime 语义。
- 不删除或移动 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`.egg-info` 或其它 ignored 本地产物。
- 不归档或修改其它 active change；只在导航文档中说明如何判断 active change 状态。

## Decisions

### Decision 1: 新增 `docs/agent_navigation.md`，不把导航内容塞进 README 或 AGENTS

`docs/agent_navigation.md` 将作为 AI/维护者修改前的快速导航。AGENTS 只保留一条指针，README 只在需要时引用，不承担完整路由表。

理由：README 已经较长，继续加入任务路由和误读清单会稀释 quickstart；AGENTS 是操作规则，不适合维护完整结构。独立导航文档可以短、集中、可测试。

备选方案是直接扩写 AGENTS。该方案入口更显眼，但会让 AGENTS 重新承担目录清单职责，和现有“本文件不维护完整目录清单”的约束冲突。

### Decision 2: 导航文档采用固定章节，而不是自由说明

导航文档至少包含：

- 当前状态检查顺序。
- 权威来源优先级。
- 任务路由表。
- 常见误读清单。
- 修改前检查清单。
- 验证命令选择表。

理由：AI 需要稳定 anchor，架构检查也需要可验证标记。自由散文更容易漏掉 generated artifacts、active change、virtual config 或本地产物边界。

备选方案是在 inventory 中新增一段“AI 注意事项”。该方案能少一个文件，但 inventory 已偏审计台账，不适合作为修改前的第一阅读入口。

### Decision 3: 权威来源优先级显式化

导航文档必须声明优先级：用户当前请求、AGENTS 操作规则、active OpenSpec change、当前 `openspec/specs/`、README/docs workflow、源码与测试、OpenSpec archive/历史报告/本地产物。active change 需要通过 `openspec list --json` 和 `openspec status --change <name>` 判断，不能只看目录是否存在。

理由：当前项目可能同时存在已完成未归档 change、archive 历史和未提交实现。显式优先级可以减少 AI 把历史或生成物当成当前契约的概率。

备选方案是只要求“读 README 和 OpenSpec”。该方案过宽，不能解决多份文档冲突时的选择问题。

### Decision 4: 任务路由表连接文档、模块和验证

导航文档将把常见变更类型映射到先读文件、主要代码区域和检查命令。例如模型改动路由到 `models/`、registry 和 forward/config tests；batch/data contract 路由到 dataset、`engine.batch`/runtime；CLI 路由到 `src/kd_sensing/cli/`、`pyproject.toml` 和 CLI help tests。

理由：这比完整目录树更有用，能直接帮助 AI 确定下一步调查路径，同时不重复维护全量项目结构。

备选方案是生成完整目录说明。该方案容易漂移，也会和 README/Inventory 的文档边界冲突。

### Decision 5: 架构边界测试只检查关键标记和生命周期链接

测试应检查 `docs/agent_navigation.md` 存在，并包含关键概念：OpenSpec 状态检查、权威来源优先级、任务路由、generated metadata、ignored runtime artifacts、virtual configs、retired research lines、验证命令和 `kd_mm_beam`。测试还应检查 `AGENTS.md` 和 `docs/project_surface_inventory.md` 引用或分类该文档。

理由：这能防止导航文档消失或退化，同时避免把测试写成文档全文快照。

备选方案是完全不测文档。该方案短期省事，但该 change 的价值正是防漂移。

### Decision 6: 不自动清理本地产物，只说明边界

本 change 不删除 `src/kd_sensing.egg-info/`、`.pytest_cache/`、`outputs/`、`logs/` 或数据目录。导航文档只说明这些路径的角色、是否可作为权威来源，以及需要清理时应走现有 manifest 或显式确认流程。

理由：用户当前需求是设计防误解方案，不是清理工作树。自动删除本地产物会扩大风险，并和现有产物边界约束冲突。

备选方案是顺手删除 generated metadata。该方案可能让工作树更干净，但会混淆“文档导航 proposal”和“本地产物清理”两个问题。

## Risks / Trade-offs

- [Risk] 新增文档成为又一个需要维护的入口。→ Mitigation：文档保持短导航，不重复 README/OpenSpec 细节，并通过 inventory 分类说明生命周期。
- [Risk] 测试过度约束文字导致维护成本高。→ Mitigation：只检查关键标记和文档引用，不检查段落原文。
- [Risk] 导航文档与项目真实结构逐渐漂移。→ Mitigation：任务路由聚焦模块类别和权威文件，而不是完整文件清单；新增入口/配置仍由 inventory 和架构边界测试维护。
- [Risk] AI 仍然只读当前打开的 generated 文件。→ Mitigation：AGENTS 增加指向导航文档的修改前入口，导航文档显式点名 `.egg-info`、`outputs/`、`logs/`、`dataset/` 等边界。
- [Risk] active change 已完成但未归档时产生双重解释。→ Mitigation：导航文档要求同时查看 `openspec list --json`、`openspec status --change <name>`、tasks 和工作树状态，再判断是否继续、归档或新建 change。

## Migration Plan

1. 新增 `docs/agent_navigation.md`，填入固定章节和当前项目路由表。
2. 在 `AGENTS.md` 的基本原则或 OpenSpec 部分加入简短指针。
3. 在 `docs/project_surface_inventory.md` 的文档生命周期分类中加入 `docs/agent_navigation.md`。
4. 扩展 `tests/test_architecture_boundaries.py`，验证导航文档关键标记、AGENTS 指针和 inventory 分类。
5. 运行 `openspec validate clarify-ai-maintainer-navigation --strict`。
6. 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

回滚策略：删除新增导航文档，移除 AGENTS/inventory 引用和对应测试，即可回到当前文档结构；该 change 不产生 runtime migration。

## Open Questions

- 导航文档是否应被 README 末尾的项目文档索引引用，还是只通过 AGENTS 和 inventory 暴露即可？
- 是否需要把“当前 active change 已完成但未归档”的处理建议写成单独小节，还是归入当前状态检查清单？
