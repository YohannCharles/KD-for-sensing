## Context

本次审视显示项目已经具备较强的架构自我保护能力：README 明确当前支持入口，`project-architecture` 与 `project-surface-cleanup` specs 约束了包结构、退役路线、运行产物边界和轻量导入，`tests/test_architecture_boundaries.py` 已经用 allowlist/guardrail 防止旧入口回流。

新的维护压力来自规模增长和并行研究线叠加：

- `src/kd_sensing` 约 6.2 万行，`tests` 约 1.56 万行，配置 YAML 约 105 个。
- AST 扫描显示仍有多个长函数和长类，例如 `engine.trainer._train_inner` 316 行、`engine.mmw_town_gps_v2.run_mmw_town_gps_v2` 277 行、BeamBench AE+GPS 两个训练函数 200 行以上、`DeepSense6GDataset` 约 1099 行、`MMWDataset` 约 592 行。
- 测试文件普遍手写 `ROOT/SRC` 和 `sys.path.insert` bootstrap，缺少集中 pytest 配置；这会让新测试容易复制旧模式，也会让未来的 import boundary 检查更难维护。
- `configs/fusion/` 根目录已有 allowlist guard，但实验子目录、root 复现文档、推荐健康检查命令和 inventory 之间仍主要依赖人工同步。
- 当前还有活跃 `add-gps-query-jepa-pooling` change，已经触碰 JEPA 模型、modular model、run metadata、配置和测试。本 change 需要避免与其源码范围冲突。

## Goals / Non-Goals

**Goals:**

- 建立项目级健康护栏，使“热点继续膨胀”“测试 bootstrap 重复”“配置/文档支持面漂移”能被 focused tests 捕捉。
- 在 inventory 中记录下一批优化热点、拆分方向和暂缓理由，让后续实现从可审计清单推进。
- 集中 pytest bootstrap，并补充 pytest 基础配置，使测试入口更一致，减少每个测试文件手写路径逻辑。
- 为实验配置和文档支持面增加机器可检查的 inventory/allowlist，而不是只靠 README 说明。
- 保持训练、评估、数据加载、模型输出、CLI 行为和本地产物边界不变。

**Non-Goals:**

- 不在本 change 中重写 `DeepSense6GDataset`、`MMWDataset`、BeamBench AE+GPS 或训练主循环的业务逻辑。
- 不引入新的强制 runtime dependency，不要求用户安装额外工具才能运行现有训练/评估。
- 不恢复 KD、HiST/Hist、Top8 selector、GPS residual、camera residual 或其它退役路线。
- 不删除、迁移或压缩 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 历史权重。
- 不把 active JEPA change 的模型/配置实现纳入本 change 的重构范围。

## Decisions

### Decision 1: 先建 guardrail，再做大拆分

第一版只新增健康检查、inventory 和少量测试基础设施整理；长函数/长类只进入“已知热点”清单和预算，不在同一 change 中进行大规模业务拆分。

替代方案是直接重构最大的 dataset、trainer 和 BeamBench workflow。该方案能立即减少文件长度，但会同时触碰数据契约、训练循环、checkpoint/metadata 和真实数据路径，容易和当前活跃 JEPA change 以及本地实验状态互相干扰。先建 guardrail 能把后续拆分边界固定住，再逐项实施。

### Decision 2: 静态健康检查使用标准库 AST 和现有 pytest

健康检查优先使用 Python 标准库 `ast`、`pathlib` 和现有 pytest，而不是第一步引入 ruff/mypy/pre-commit 作为硬依赖。检查内容包括超长函数/类清单、已知 hotspot allowlist、测试 bootstrap 漂移、脚本/config 文档引用一致性和当前支持面 inventory。

替代方案是直接引入 ruff、mypy 或 pre-commit。它们长期有价值，但会引入环境安装、规则基线和历史代码批量修复问题。当前项目的第一目标是减少架构漂移风险，不是一次性格式化或类型化全仓库。

### Decision 3: 已知热点允许存在，但必须被命名和约束

对当前已经存在的超长函数/类不立即失败；它们必须进入 `docs/project_surface_inventory.md` 或等价 inventory，记录文件、符号、问题类型、推荐拆分方向和当前暂缓原因。新增或明显扩大的热点必须更新 inventory，或拆出窄模块。

替代方案是设置全仓统一行数上限。这个方案简单，但会让已有历史热点导致测试长期红灯，最终迫使大家提高阈值。显式 hotspot inventory 更适合研究型仓库：允许已知债务存在，但禁止它悄悄增加。

### Decision 4: 测试 bootstrap 集中到 `tests/conftest.py`

新增 shared pytest bootstrap，把 `src/` 加入 import path 的逻辑放到 `tests/conftest.py`。后续测试文件不再复制 `ROOT = ...`、`SRC = ...`、`sys.path.insert(...)` 片段；已有文件可分批迁移，架构边界测试先拒绝新增重复模式或要求例外记录。

替代方案是只依赖 editable install。虽然 README 推荐 `python -m pip install -e .`，但当前测试经常直接在源码树运行，并且一些 import-boundary probe 故意在子进程中控制 `sys.path`。集中 bootstrap 兼顾本地便利和边界测试可控性。

### Decision 5: 配置与文档支持面延续 allowlist 模式

根目录 fusion YAML 已有 allowlist；本 change 将同一思想扩展到实验配置分区和 root 文档：长期推荐入口、实验复现配置、历史/研究笔记和本地产物说明都需要在 inventory 中归类。测试只检查“当前支持面是否有清晰归属”，不要求删除历史文档。

替代方案是把所有历史/研究文档迁入 archive。该方案会产生较大文档移动和引用更新成本，也可能掩盖仍有价值的复现实验记录。先分类和检查引用更稳。

### Decision 6: 健康检查分层而不是一个巨型命令

文档记录分层命令：

- OpenSpec：`openspec validate strengthen-project-health-guardrails --strict`
- 架构边界：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- CLI/config smoke：`conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`
- 触碰训练、数据、诊断或具体 workflow 时追加对应 focused tests

替代方案是新增一个长期 console script 执行全部检查。该方案会扩大公开入口表面，也容易把慢测试和环境依赖混在一起。分层命令更符合当前 README/OpenSpec 的验证方式。

## Risks / Trade-offs

- [Risk] 静态 guardrail 过严会阻塞正常研究迭代。→ Mitigation：第一版只对新增漂移和 inventory 不一致严格失败；既有热点通过显式清单管理。
- [Risk] 只用 AST/pytest 无法替代 lint/type check。→ Mitigation：把 ruff/mypy/pre-commit 留作后续可选质量层；本 change 先解决当前最明显的结构漂移问题。
- [Risk] 集中 `tests/conftest.py` 后，个别子进程 import-boundary 测试仍需要手动 `sys.path`。→ Mitigation：允许 architecture-boundary probe 内部显式控制 `sys.path`，但普通测试文件不再复制 bootstrap。
- [Risk] 配置/文档 allowlist 维护成本上升。→ Mitigation：inventory 必须解释分类边界；新增配置或文档时只需补一处分类和对应测试期望。
- [Risk] 活跃 JEPA change 同时修改配置和测试，可能让 inventory 数量变化。→ Mitigation：本 change 不编辑 JEPA 源码；实现时若 JEPA change 已合入，则按其最终配置集合更新 inventory。

## Migration Plan

1. 新增 `project-health-guardrails` spec，并为 `project-architecture`、`project-surface-cleanup` 添加 delta requirement。
2. 新增或更新测试基础设施：`tests/conftest.py`、pytest 配置、架构边界中的健康 guardrail。
3. 更新 `docs/project_surface_inventory.md`，加入第二批热点、实验配置分区和 root 文档分类。
4. 分批移除普通测试文件中重复的 `sys.path.insert` bootstrap；保留 import-boundary 子进程 probe 的显式路径控制。
5. 运行 OpenSpec 和 focused tests。

Rollback 方式是删除本 change 新增的测试基础设施、inventory 段落和 spec delta；由于不改变训练/数据/模型 runtime 行为，rollback 不需要迁移 checkpoint、cache 或本地输出。

## Open Questions

- 第一版是否只做 AST/pytest guardrail，还是同时把 ruff 加入 `dev` extra 但不作为默认验证命令。
- 热点行数预算是否按 symbol 类型区分，例如 orchestration 函数 160 行、dataset 类 500 行、facade 200 行，还是先只输出报告再逐步收紧。
- root 研究文档是否需要新增 `docs/archive/` 分类，还是继续保留在根目录但用 inventory 标记生命周期。
