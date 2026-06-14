# AI / Maintainer Navigation

本文件是 AI agent 和维护者在非平凡改动前的薄导航层，用来判断权威来源、当前状态、任务路由、常见误读和验证命令。它不替代 README 的 quickstart、AGENTS 的操作规则、OpenSpec specs 的需求契约，也不维护完整源码目录清单。

## 当前状态检查顺序

1. 先读用户当前请求和本轮对话中的限制；它们只约束本次工作，不自动改写长期契约。
2. 读 `AGENTS.md`，确认命令环境、OpenSpec、文档边界和本地产物边界；所有项目 Python 命令使用 `conda run -n kd_mm_beam ...`。
3. 检查 active change 状态：运行 `openspec list --json`，再对目标 change 运行 `openspec status --change <change>`；已完成但未归档的 change 仍可能影响当前工作树解释。
4. 读 active change 的 `proposal.md`、`design.md`、`tasks.md` 和 `specs/**/*.md`；没有 active change 时，先看 `docs/project_surface_inventory.md` 的 OpenSpec capability lifecycle 分类，再读 `openspec/specs/` 中对应 specs。
5. 对每个 capability 先判定 lifecycle：`current` 才能作为当前需求契约或推荐入口；`supporting` 只能理解为当前 workflow 消费的 helper、metric、manifest、cleanup 或 migration guard；`retired-tombstone` 只解释为退役边界、防回流或 migration guard，不代表当前运行入口。
6. 读 README 和 `docs/` 中对应 workflow 文档，确认当前推荐入口、退役说明和验证建议。
7. 最后看源码、测试和 `git status --short`，确认实际实现、未提交改动和 ignored runtime artifacts 没有被误当作源码需求。

## 权威来源优先级

多份资料看似冲突时，按以下顺序判断：

1. 用户当前请求和显式限制。
2. `AGENTS.md` 中的操作规则、命令环境和产物边界。
3. active OpenSpec change 的 proposal/design/spec/tasks，以及 `openspec status --change <change>` 给出的状态。
4. `docs/project_surface_inventory.md` 中的 OpenSpec capability lifecycle 分类，以及当前 `openspec/specs/` 中已归档为当前需求的 specs。
5. README、`docs/` workflow、`docs/project_surface_inventory.md` 其它 inventory 内容和复现说明。
6. 源码和测试中已经存在的实现契约。
7. `openspec/changes/archive/`、历史报告、旧研究笔记、本地数据和运行产物。

OpenSpec archive、历史报告和本地产物不能覆盖当前 specs。Capability 文件名也不能覆盖 lifecycle 分类：旧能力名称如果被标为 `retired-tombstone`，就只能作为墓碑解释；标为 `supporting` 时，也必须继续查 README、inventory 或 current workflow spec 来确认实际推荐入口。当前打开文件不等于项目权威入口，尤其不要从 generated metadata 或输出目录反推当前支持面。

## 任务路由表

| 改动类型 | 先读什么 | 主要修改区域 | 常用验证 |
| --- | --- | --- | --- |
| 模型 / forward / registry 暴露 | 对应 OpenSpec specs、README 当前模型说明、`docs/project_surface_inventory.md` 热点和退役边界 | `src/kd_sensing/models/`、`src/kd_sensing/registries.py`、默认组件和 forward 输出消费处 | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，再追加模型/forward focused tests |
| 数据与 batch contract | dataset/modality contract specs、README 数据边界、inventory 中 `engine.batch` 热点 | `src/kd_sensing/data/`、`src/kd_sensing/engine/batch.py`、shared runtime、相关 dataset tests | 相关 dataset/batch focused tests；避免读取真实 `dataset/` |
| 配置和 virtual config | 配置生命周期 specs、README 配置章节、inventory 配置生命周期分类 | `src/kd_sensing/config/`、`configs/`、canonical recipe / virtual config 生成规则 | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` 和架构边界测试 |
| CLI / scripts 入口 | README 主要入口、inventory 脚本入口 allowlist、`pyproject.toml` console scripts | `src/kd_sensing/cli/`、`scripts/` allowlist、`pyproject.toml` | CLI help smoke、`tests/test_cli_help.py`、架构边界测试 |
| 输出产物 / cache / cleanup | README 数据和产物边界、inventory 本地产物分类、cleanup manifest workflow | `src/kd_sensing/utils/runtime_output_layout.py`、diagnostic / cleanup CLI；默认输出在 ignored `outputs/` | 不写入 `outputs/` 或 `logs/` 的单元测试；必要时只生成 dry-run manifest |
| 诊断 / viewer / visual analysis | README Viewer Manifest、诊断 specs、inventory viewer manifest 热点 | `src/kd_sensing/diagnostics/`、`src/kd_sensing/cli/export_viewer_manifest.py`、诊断配置 | `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py -q` 和 CLI help |
| OpenSpec artifact | active change 的 proposal/design/spec/tasks、inventory lifecycle、当前 specs、`openspec status` | `openspec/changes/<change>/` 或 `openspec/specs/` | `openspec validate <change> --strict`，必要时 `openspec status --change <change>` |
| 文档生命周期 | AGENTS 文档边界、README 文档索引、OpenSpec capability lifecycle、inventory 文档生命周期分类 | README、`AGENTS.md`、`docs/*.md`、OpenSpec 文档 | 架构边界测试；检查不把历史、supporting 或退役墓碑路线写成当前推荐入口 |

## 常见误读清单

- generated metadata：`src/kd_sensing.egg-info/SOURCES.txt`、`entry_points.txt`、`dependency_links.txt` 等是 packaging 生成元数据，不是源码结构或入口权威；判断入口时看 `pyproject.toml`、`src/kd_sensing/`、README 和 OpenSpec。
- OpenSpec capability lifecycle：先看 `docs/project_surface_inventory.md` 的 `current`、`supporting`、`retired-tombstone` 分类。`supporting` 不等于 standalone 当前入口；`retired-tombstone` 只保留退役、防回流或 migration guard 说明。
- ignored runtime artifacts：`outputs/`、`outputs/cache/`、`logs/`、legacy 根 `cache/`、`.pytest_cache`、`__pycache__`、`.pyc`、TensorBoard 文件和 checkpoint 是本地运行产物，不能自动纳入源码变更，也不能作为当前支持入口证据。
- pytest cache：`.pytest_cache/v/cache/lastfailed` 只记录本地上一次 pytest 状态，可能已经过期；真实红点以当前测试文件和实际 `pytest` 命令结果为准。
- 本地数据：`dataset/` 是本地输入，默认只允许源码里保留 `dataset/.gitkeep`；测试和文档改动不得读取真实数据来证明契约。
- OpenSpec archive：`openspec/changes/archive/` 是历史记录，只能解释演进过程；未跟踪 archive 目录、已归档但未提交的 change 或 archive 中的新 spec 不能当作 active change。当前需求以 active change、lifecycle inventory 和当前 `openspec/specs/` 为准。
- retired research lines：旧 KD、HiST/Hist、Top8 selector、GPS residual、camera residual、Raymobtime s008、CRAF/MARF/G2D/Multimodal-NF 等只能作为历史或 migration guard 说明出现，不得用兼容 wrapper、旧 CLI 或实体 YAML 恢复为当前入口。
- virtual configs：部分 `configs/fusion/*.yaml` 路径可能由配置加载器生成，没有实体 YAML；先查 README、inventory 和 config specs。不得让 virtual config 接管退役 `logits_kd` / `rkd` / old residual 路径。
- active change 状态：目录存在不等于正在实施，任务全勾选也不等于已经归档；同时看 `openspec list --json`、`openspec status --change <change>`、tasks 和工作树状态。
- 当前打开文件：IDE 打开的 generated metadata、历史报告或输出摘要只是局部上下文，不能替代 README、AGENTS、OpenSpec 和源码测试。
- 语义冲突：如果同一 current spec 内部同时把某路线写成 active mainline 和 retired/supporting，先把它视为规格漂移；通过 OpenSpec 清理 change 收敛到 current/supporting/retired lifecycle，而不是任选一段当事实。

## 修改前检查清单

- 本次改动是否非平凡、涉及架构、训练流程、数据契约、配置兼容或公共入口；如果是，先确认或创建 OpenSpec change。
- 是否已经读过 active change 的 proposal/design/spec/tasks 和相关当前 specs。
- 是否确认要修改的文件不是 generated metadata、ignored runtime artifacts、本地数据或历史 archive。
- 是否确认不新增旧入口、兼容聚合层、退役研究线实体配置或绕过 `src/kd_sensing` 包结构的运行方式。
- 是否知道最小 focused tests；无法运行时，需要在最终说明中写清楚原因和剩余风险。

## 验证命令选择表

| 触碰范围 | 推荐命令 |
| --- | --- |
| OpenSpec change | `openspec validate <change> --strict` |
| 架构、文档生命周期、入口 allowlist、本地产物边界 | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` |
| CLI 或 console script | `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`，并按需运行对应 `--help` |
| 配置解析、virtual config、migration guard | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` |
| 诊断、viewer manifest、模态可视化 | `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py -q` |
| 训练、数据、模型 forward 或 shared runtime | 先跑对应 focused tests；高风险改动再考虑 `conda run -n kd_mm_beam pytest -q` |

所有验证都应避免把新生成的 `dataset/` 内容、`outputs/`、`logs/`、cache、checkpoint 或 `.egg-info` 变更纳入源码提交。
