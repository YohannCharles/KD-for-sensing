# AI / Maintainer Navigation

本文件是 AI agent 和维护者在非平凡改动前的薄导航层，用来判断权威来源、当前状态、任务路由、常见误读和验证命令。它不替代 README 的 quickstart、AGENTS 的操作规则、OpenSpec specs 的需求契约，也不维护完整源码目录清单。

## 当前一屏摘要

- 当前主线：多模态少样本跨场景 beam prediction，重点围绕 Image+GPS JEPA query-pool、缺失模态本地实验、MMW/CSI 诊断和当前保留的 baseline/reproduction workflow。
- 推荐入口：训练、评估、预处理和诊断优先使用 `pyproject.toml` 声明的 `kd-sensing-*` console scripts；Scene31/Scene31-34 队列、表格和结论脚本只作为 local/manual 或 research diagnostic surface。
- 研究预览闭环：长跑、手动拼表或 claim 更新前可先运行 `conda run -n kd_mm_beam kd-sensing-research-preview --no-resources`；它默认只生成 preview manifest、dashboard HTML、静态 evidence QA 和 budget manifest，不启动训练、不读取真实 `dataset/`、不加载 checkpoint。
- 渐进加载入口：任务细节优先按 `docs/agent_context/README.md` 选择 scoped context；spec/config/claim 快速扫视用 `docs/agent_context/atlas.md`，不要把 atlas 当成需求契约。
- 跨工具 agent 入口：Claude、Copilot、Cursor、Kiro 和 Project Knowledge 适配文件只做薄引用；`docs/current_research_brief.md` 是研究方向简报，不替代 `docs/result_claims_registry.md` 或 `docs/experiment_protocols.md`。
- 绝对退役边界：旧 KD、HiST/Hist、Top8 selector、GPS residual、camera residual、BGAM、viewer manifest、Raymobtime s008、AMR-Net_gps_image 和 JEPA-MSAC 不得通过兼容 wrapper、旧 CLI、实体 YAML 或 package facade 恢复。
- 必读入口：`AGENTS.md`、本文件、`docs/project_surface_inventory.md`、`docs/maintainer_context_index.yaml`、README 和目标 OpenSpec spec；有 active change 时先读对应 proposal/design/tasks/specs。
- 快速验证：常规无数据入口用 `make verify-quick`；CLI/config 变更追加 `make verify-cli-config`；脚本/CLI 语法检查用 `make verify-compile`。底层命令仍是 `openspec validate --all --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 和 CLI/config focused tests。
- 判断口径：优先看 tracked/source 和 current lifecycle，不要从 generated metadata、ignored outputs、pytest cache、当前打开文件或历史 archive 反推当前支持面。

## 当前状态检查顺序

1. 先读用户当前请求和本轮对话中的限制；它们只约束本次工作，不自动改写长期契约。
2. 读 `AGENTS.md`，确认命令环境、OpenSpec、文档边界和本地产物边界；所有项目 Python 命令使用 `conda run -n kd_mm_beam ...`。
3. 检查 active change 状态：运行 `openspec list --json`，再对目标 change 运行 `openspec status --change <change>`；已完成但未归档的 change 应先归档或记录 deferral，否则不要把它当成仍在实施的需求。归档后的快速 OpenSpec 验证优先复制 `openspec validate --all --strict` 或 current spec validation，不复制 archive change 名。
   - 收口前可运行只读 preflight：`conda run -n kd_mm_beam kd-sensing-project-surface-doctor --scope closeout --format markdown --fail-on none`。它只报告 active/complete/archive/untracked change 和 dirty worktree 分类，不会 archive、reset、删除或移动任何文件。
   - 若同时看到 active change 删除和 `openspec/changes/archive/<date>-<name>/` 新增，把它当作同一个 closeout 状态处理；先确认验证和提交边界，不要把新 archive 当成当前需求。
4. 读 `docs/project_surface_inventory.md`，用 inventory 定位 lifecycle、入口分类、root fusion YAML、experiment config family、脚本 lifecycle、热点说明和历史 caveat；`docs/maintainer_context_index.yaml` 只保留退役 token、task route、验证命令等最小结构化事实，不维护完整源码目录清单、入口 allowlist 或 prose 镜像。
5. 按任务读取 `docs/agent_context/` 中的 scoped context；只在需要扫视 spec/config/claim owner、lifecycle、focused tests 和 caveat 时读取 `docs/agent_context/atlas.md`。
6. 需要快速判断研究主线时可读 `docs/current_research_brief.md`；它只给方向和 gate，不替代 claim registry、experiment protocols、mainline catalog、experiment matrix 或 OpenSpec。
7. 读 active change 的 `proposal.md`、`design.md`、`tasks.md` 和 `specs/**/*.md`；没有 active change 时，先看 inventory 的 OpenSpec capability lifecycle 分类，再读 `openspec/specs/` 中对应 specs。
8. 对每个 capability 先判定 lifecycle：`current` 才能作为当前需求契约或推荐入口；`supporting` 只能理解为当前 workflow 消费的 helper、metric、manifest、cleanup 或 migration guard；`retired-tombstone` 只解释为退役边界、防回流或 migration guard，不代表当前运行入口。
9. 读 README 和 `docs/` 中对应 workflow 文档，确认当前推荐入口、退役说明和验证建议。
10. 最后看源码、测试和 `git status --short`，确认实际实现、未提交改动、未分类 `scripts/`/root runbook 和 ignored runtime artifacts 没有被误当作源码需求；`.codegraph/daemon.pid`、socket、db、cache 和 log 只能作为本地工具状态处理。

## 权威来源优先级

多份资料看似冲突时，按以下顺序判断：

1. 用户当前请求和显式限制。
2. `AGENTS.md` 中的操作规则、命令环境和产物边界。
3. active OpenSpec change 的 proposal/design/spec/tasks，以及 `openspec status --change <change>` 给出的状态。
4. `docs/project_surface_inventory.md` 中的 OpenSpec capability lifecycle 分类、入口说明、热点 rationale 和历史 caveat。
5. `docs/maintainer_context_index.yaml` 中无法从 pyproject、OpenSpec、真实路径或 inventory 推导的最小结构化事实。
6. README、`docs/` workflow、`docs/project_surface_inventory.md` 其它 inventory 内容和复现说明。
7. 源码和测试中已经存在的实现契约。
8. `openspec/changes/archive/`、历史报告、旧研究笔记、本地数据和运行产物。

OpenSpec archive、历史报告和本地产物不能覆盖当前 specs。Capability 文件名也不能覆盖 lifecycle 分类：旧能力名称如果被标为 `retired-tombstone`，就只能作为墓碑解释；标为 `supporting` 时，也必须继续查 README、inventory 或 current workflow spec 来确认实际推荐入口。当前打开文件不等于项目权威入口，尤其不要从 generated metadata、测试常量或输出目录反推当前支持面；先用 inventory 和 current specs 确认 lifecycle。

`docs/maintainer_context_index.yaml` 和 `docs/project_surface_inventory.md` 职责不同：前者只是最小结构化事实清单；后者保留解释性审计、历史上下文、caveat 和暂缓原因。架构边界测试应验证 pyproject、真实路径、tracked files、current config glob、retired token 语境和禁止 import 这些结构事实，不逐字复制文档段落。二者或 README、导航文档、OpenSpec specs 之间出现看似冲突时，先视为治理漂移，通过 OpenSpec change 同步 inventory、最小索引和对应 specs，不要任选一处作为事实。

## 任务路由表

Scoped context 优先作为按需入口，详细 rationale 仍回到 inventory、README 和 OpenSpec specs。

| Route id | Scoped context | 常见触发 |
| --- | --- | --- |
| `model` | `docs/agent_context/models.md` | 模型、forward、registry、baseline、组件扩展 |
| `data` | `docs/agent_context/data.md` | dataset、batch contract、modality profile、split |
| `config` | `docs/agent_context/configs.md` | YAML、virtual config、canonical recipe、migration guard |
| `cli` | `docs/agent_context/cli.md` | console scripts、包内 CLI、`scripts/` 入口 |
| `diagnostics` | `docs/agent_context/diagnostics.md` | run index、JEPA/GPS benchmark、doctor、paper export |
| `openspec` | `docs/agent_context/openspec.md` | proposal/spec/tasks/archive、complete active change 收口 |
| `documentation` | `docs/agent_context/documentation.md` | README、AGENTS、inventory、导航、文档健康 |
| `claims` | `docs/agent_context/claims.md` | result claim registry、论文表格、provenance |
| `atlas` | `docs/agent_context/atlas.md` | spec/config/claim owner、lifecycle、focused tests 快速扫视 |

| 改动类型 | 先读什么 | 主要修改区域 | 常用验证 |
| --- | --- | --- | --- |
| 模型 / forward / registry 暴露 | `model-architecture-extension-contract`、`modular-sequence-model`、`component-registry`、README 当前模型说明、inventory 热点和退役边界 | `src/kd_sensing/models/`、`src/kd_sensing/registries.py`、默认组件、`engine.batch` / `engine.runtime` forward 输出消费处 | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，再追加模型/forward focused tests |
| 数据与 batch contract | 索引的 `data_batch_contract` 路由、dataset/modality contract specs、README 数据边界、inventory 中 `engine.batch` 热点 | `src/kd_sensing/data/`、`src/kd_sensing/engine/batch.py`、shared runtime、相关 dataset tests | 相关 dataset/batch focused tests；避免读取真实 `dataset/` |
| 配置和 virtual config | 配置生命周期 specs、README 配置章节、inventory 配置生命周期分类 | `src/kd_sensing/config/`、`configs/`、`canonical.py` 中的 canonical recipe / virtual config 生成规则 | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` 和架构边界测试 |
| CLI / scripts 入口 | README 主要入口、`pyproject.toml` console scripts、inventory 脚本入口分类 | `src/kd_sensing/cli/`、`scripts/`、`pyproject.toml`；真实 workflow 逻辑应位于 owner module/script | CLI help smoke、`tests/test_cli_help.py`、架构边界测试 |
| 输出产物 / cache / cleanup | README 数据和产物边界、inventory 本地产物分类、cleanup manifest workflow | `src/kd_sensing/utils/runtime_output_layout.py`、diagnostic / cleanup CLI；默认输出在 ignored `outputs/` | 不写入 `outputs/` 或 `logs/` 的单元测试；必要时只生成 dry-run manifest |
| 诊断 / visual analysis / benchmark | 诊断 specs、JEPA visual analysis、GPS shortcut benchmark、inventory 诊断热点 | `src/kd_sensing/diagnostics/`、`src/kd_sensing/cli/jepa_visual_analysis.py`、`src/kd_sensing/cli/jepa_gps_shortcut_benchmark.py`、诊断配置 | `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q` 和 CLI help |
| research preview / evidence QA / budget manifest | `research-run-preview-loop` active/current spec、diagnostics 和 claims scoped context、inventory package CLI 分类 | `src/kd_sensing/diagnostics/research_run_preview.py`、`src/kd_sensing/cli/research_preview.py`、README/experiment matrix | `conda run -n kd_mm_beam pytest tests/test_research_run_preview.py tests/test_cli_help.py -q` |
| OpenSpec artifact | active change 的 proposal/design/spec/tasks、inventory lifecycle、当前 specs、`openspec status` | `openspec/changes/<change>/` 或 `openspec/specs/` | `openspec validate <change> --strict`，必要时 `openspec status --change <change>` |
| 文档生命周期 | 索引的 `documentation_lifecycle` 路由、AGENTS 文档边界、README 文档索引、OpenSpec capability lifecycle、inventory 文档生命周期分类 | README、`AGENTS.md`、`docs/*.md`、OpenSpec 文档 | 架构边界测试；检查不把历史、supporting 或退役墓碑路线写成当前推荐入口 |

内部源码和测试默认导入真实 owner 模块：例如 objective metadata 使用 `kd_sensing.engine.objectives.metadata`，BeamBench Image AE+GPS 使用 `image_ae_gps_training.py` / `image_ae_gps_paper_split.py` 等具体 owner，fusion 测试使用 `fusion.cls_token_transformer` 或 `fusion.token_transformer`。不要为了省 import 行恢复 package-level re-export、lazy export 或旧聚合 facade。

跨工具 agent 适配文件（例如 `CLAUDE.md`、`.github/copilot-instructions.md`、`.cursor/rules/*.mdc`、`.kiro/steering/*.md` 和 Project Knowledge 模板）只允许保留短引用、工具加载差异和必要边界提醒；不得复制完整任务路由、完整 OpenSpec requirement、完整退役清单或完整 claim 表。重复 agent 错误先记录到 `docs/agent_memory_ledger.md` 的候选清单，人工确认或后续 OpenSpec change 后再沉淀到长期文档。只读角色见 `docs/readonly_agent_roles.md`，它们只能输出建议，不直接写文件、不启动训练、不清理本地产物。

新增 current mainline、paper reproduction、benchmark 或诊断 workflow 时，必须同步四层文档：`docs/mainline_model_catalog.md` 记录当前事实行，`docs/experiment_protocols.md` 记录参数口径，`docs/result_claims_registry.md` 记录 claim/provenance，`docs/experiment_matrix.md` 记录 quickstart 命令和关键 caveat。若该实验改变主线取舍、形成复盘结论或暴露新创新线索，还应补 `docs/mainline_experiment_history.md`。若该 workflow 有明确名称或专用入口，还应在 inventory 或 focused 架构测试中登记 owner module/script、responsibility、output boundary 和必要 retired route guard。

Scene31 night-grid / next-round / BC / beamsoft weak / funnel / magic overnight 属于 manifest-backed local/manual workflow：修改 `configs/scene31/` 或 `scripts/run_scene31_*.sh` 前，先查 active/archived OpenSpec 状态、`scene31-next-round-experiment-workflow` spec、inventory 的 config/script 分类、真实 tracked YAML/runner 清单和 `tests/test_scene31_next_round.py` / `tests/test_architecture_boundaries.py`。源码只保留 manifest、base config、generator、必要保留的 local/manual overlay 和薄 runner/helper；generator-backed 实体 YAML 需本地生成后再走 `kd-sensing-train --config <generated-yaml>`，fresh eval/analysis 产物只写 ignored output/log roots。不要把 generated YAML、ignored outputs 或 completed archive change 误当作 current source requirement。

## 热点右尺寸化决策矩阵

修改已登记 hotspot、接近预算的 workflow、dataset、diagnostic module 或 facade 前，先读取 inventory 中的 architecture sizing baseline、hotspot rationale 和当前 focused tests。不要把 Python 文件数、function 数或 import 数机械解释成“所有大文件都要拆”：这些只是趋势信号；每次变更先判断动作属于 `split`、`consolidate`、`monitor`、`accepted-size`、`hard-budget`、源码窄修复或 `keep-and-test`。

| 场景 | 默认动作 | 需要确认 |
| --- | --- | --- |
| 公开 CLI/import facade 超预算或吸收 helper | `hard-budget` / `owner-facade`，实现移回窄模块 | `enforcement: hard-fail`、public import/CLI smoke、无 helper 回流 |
| 业务 workflow、dataset 或 diagnostic 稍超预算 | 按 rationale 和 `headroom_lines` 判断 `split`、`monitor` 或 `accepted-size` | focused tests、headroom 是否仍足够、是否有下一步 split target |
| 单调用点包装类、只为减行数的 helper、重复 `utils` 聚合 | `consolidate` 或登记 `merge-candidate` | owner 清晰、合并不绕过 `src/kd_sensing` 包结构、不恢复旧入口 |
| 小而内聚的 loss/model/helper | `keep-and-test` 或 `right-size-accepted` | 保留理由、focused tests、未来增长触发条件 |

`right-size-accepted` 不是永久豁免；它只表示当前尺寸比继续拆分更可维护，仍必须保留 validation commands、accepted rationale 和 rollback note。`merge-candidate` 也不是搁置标签；它必须写明 owner、`consolidation_targets`、public surface policy 和验证命令。

## Remediation Wave Campaign

当用户明确要求完整修复热点架构并接受高风险时，按 remediation wave 实施，而不是把多个热点混成单次不可定位的大改。开始前确认 active OpenSpec change，读取 inventory，列出 wave 顺序、目标文件、计划动作、公开 surface 策略、focused tests 和 rollback 条件。

当前 `right-size-project-architecture` campaign 的 remediation wave 顺序记录在 inventory 和对应 OpenSpec change 中：Wave 0 是治理 schema、architecture sizing baseline、inventory、AI 导航和架构边界测试；Wave 1 是 BeamBench Image AE+GPS；Wave 2 是 DeepSense6G/MMW dataset 与 trainer；Wave 3 是 evaluation pass 和 diagnostics 二级热点；Wave 4 是 JEPA benchmark accepted owner；Wave 5 是 consolidation/import 面收口与 keep-and-test。当前 IDE 打开的文件只作为局部线索；即使打开 loss、model、diagnostics 或 dataset 文件，也必须放回 remediation wave、public surface policy、merge-candidate/accepted owner 和 rollback 边界判断。

模型/forward/registry 改动先归类：

- config-only baseline：只改 YAML、canonical recipe、overlay 或 hyperparameter，优先复用 `modular_sequence`。
- component baseline：新增或替换 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 或 `HEADS`，并通过 `model.primary` 配置选择。
- whole-model exception：新增 `@MODELS.register(...)` 前必须在 active OpenSpec artifact 或 current spec 中说明原因，并补 registry build、synthetic forward、`adapt_model_output` 和 metadata tests。
- workflow/paper reproduction：官方协议、多阶段训练、feature cache 或特殊 Table 报告走 `src/kd_sensing/baselines/<family>/`、包内 CLI 或 package console script，不复制通用训练循环。

触碰 observability/reliability metadata 时，还要确认普通 baseline 可忽略新增 metadata，opt-in 模型才接收 reliability fields，并追加 difficulty/batch 或 observability-aware fusion focused tests。

## 常见误读清单

- generated metadata：`src/kd_sensing.egg-info/SOURCES.txt`、`entry_points.txt`、`dependency_links.txt` 等是 packaging 生成元数据，不是源码结构或入口权威；判断入口时看 `pyproject.toml`、`src/kd_sensing/`、README 和 OpenSpec。
- machine-readable index：`docs/maintainer_context_index.yaml` 只保存退役 token 和验证命令等最小结构化事实；它不是运行时配置、训练配置、OpenSpec requirement 全文、入口 allowlist、hotspot budget、文档短语镜像或完整源码目录清单。
- OpenSpec capability lifecycle：先看 `docs/project_surface_inventory.md` 的 `current`、`supporting`、`retired-tombstone` 分类。`supporting` 不等于 standalone 当前入口；`retired-tombstone` 只保留退役、防回流或 migration guard 说明。
- ignored runtime artifacts：`outputs/`、`outputs/cache/`、`logs/`、legacy 根 `cache/`、`.pytest_cache`、`__pycache__`、`.pyc`、TensorBoard 文件和 checkpoint 是本地运行产物，不能自动纳入源码变更，也不能作为当前支持入口证据。
- pytest cache：`.pytest_cache/v/cache/lastfailed` 只记录本地上一次 pytest 状态，可能已经过期；真实红点以当前测试文件和实际 `pytest` 命令结果为准。
- 本地数据：`dataset/` 是本地输入，默认只允许源码里保留 `dataset/.gitkeep`；测试和文档改动不得读取真实数据来证明契约。
- OpenSpec archive：`openspec/changes/archive/` 是历史记录，只能解释演进过程；未跟踪 archive 目录、已归档但未提交的 change 或 archive 中的新 spec 不能当作 active change。若 `git status --short` 同时显示 active change 删除和同名 dated archive 新增，需要把二者作为成对提交状态审计或记录 deferral。当前需求以 active change、lifecycle inventory 和当前 `openspec/specs/` 为准。
- retired research lines：旧 KD、HiST/Hist、Top8 selector、GPS residual、camera residual、BGAM、viewer manifest、仓库级 Gradio viewer、Raymobtime s008、CRAF/MARF/G2D/Multimodal-NF 等只能作为历史或 migration guard 说明出现，不得用兼容 wrapper、旧 CLI 或实体 YAML 恢复为当前入口。
- virtual configs：部分 `configs/fusion/*.yaml` 路径可能由配置加载器生成，没有实体 YAML；先查 README、inventory 和 config specs。不得让 virtual config 接管退役 `logits_kd` / `rkd` / old residual 路径。
- active change 状态：目录存在不等于正在实施，任务全勾选也不等于已经归档；同时看 `openspec list --json`、`openspec status --change <change>`、tasks 和工作树状态。
- 当前打开文件：IDE 打开的 generated metadata、历史报告或输出摘要只是局部上下文，不能替代 README、AGENTS、OpenSpec 和源码测试。
- 语义冲突：如果同一 current spec 内部同时把某路线写成 active mainline 和 retired/supporting，先把它视为规格漂移；通过 OpenSpec 清理 change 收敛到 current/supporting/retired lifecycle，而不是任选一段当事实。

## 修改前检查清单

- 本次改动是否非平凡、涉及架构、训练流程、数据契约、配置兼容或公共入口；如果是，先确认或创建 OpenSpec change。
- 是否已用 inventory、OpenSpec 和最小结构化事实定位任务路由、退役边界和最小验证命令。
- 触碰 CLI、console script 或 `scripts/` 入口时，是否已先查 `pyproject.toml`、inventory 和 OpenSpec 中的 owner module/script、responsibility、output boundary 和 retired route guard，并确认改动仍是 thin parser/IO glue 或登记的 owner 实现。
- 是否已经读过 active change 的 proposal/design/spec/tasks 和相关当前 specs。
- 是否确认要修改的文件不是 generated metadata、ignored runtime artifacts、本地数据或历史 archive。
- 是否确认不新增旧入口、兼容聚合层、退役研究线实体配置或绕过 `src/kd_sensing` 包结构的运行方式。
- 是否知道最小 focused tests；无法运行时，需要在最终说明中写清楚原因和剩余风险。

## 验证命令选择表

| 触碰范围 | 推荐命令 |
| --- | --- |
| 常规无数据 quick verify | `make verify-quick` |
| OpenSpec change | `openspec validate <change> --strict` |
| 架构、文档生命周期、入口 allowlist、本地产物边界 | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` |
| CLI 或 console script | `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`，并按需运行对应 `--help` |
| 配置解析、virtual config、migration guard | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` |
| CLI/config 聚合 smoke | `make verify-cli-config` |
| tracked scripts/package CLI Python 语法 | `make verify-compile` |
| 诊断、JEPA visual analysis、GPS shortcut benchmark | `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q` |
| research preview、静态 evidence QA、budget manifest | `conda run -n kd_mm_beam pytest tests/test_research_run_preview.py -q` |
| 训练、数据、模型 forward 或 shared runtime | 先跑对应 focused tests；高风险改动再考虑 `conda run -n kd_mm_beam pytest -q` |
| reliability-aware / observability-aware 模型 metadata | 对应模型 focused tests、difficulty/batch tests；同时覆盖普通 baseline 忽略 metadata 和 opt-in 模型接收 metadata |

所有验证都应避免把新生成的 `dataset/` 内容、`outputs/`、`logs/`、cache、checkpoint 或 `.egg-info` 变更纳入源码提交。
