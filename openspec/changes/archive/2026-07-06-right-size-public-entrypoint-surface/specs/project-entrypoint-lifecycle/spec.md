## ADDED Requirements

### Requirement: Package console scripts 必须有生命周期分类
`pyproject.toml` 中每个 `kd-sensing-*` console script MUST 被分类为 `core_workflow`、`current_diagnostic`、`paper_export`、`baseline_reproduction`、`local_manual`、`internal_only` 或 `delete`。保留为 public CLI 的入口 MUST 在 inventory、current docs 或 OpenSpec 中记录 owner module、职责、输出边界、真实数据/训练副作用边界和 focused validation。

#### Scenario: console script 分类完整
- **WHEN** 开发者检查 `pyproject.toml` 的 `[project.scripts]`
- **THEN** 每个 `kd-sensing-*` entry point MUST 在 `docs/project_surface_inventory.md` 或等价 lifecycle 文档中有分类
- **AND** 分类 MUST 说明该入口是否属于 core workflow、current diagnostic、paper/export、baseline reproduction 或其它明确状态

#### Scenario: 保留 public CLI 具备四个锚点
- **WHEN** 一个 console script 继续作为 public CLI 暴露
- **THEN** 项目 MUST 同时维护 pyproject entry point、help smoke 或无副作用 smoke、owner/output-boundary 文档和 current docs/OpenSpec 引用
- **AND** 缺少任一锚点时 implementation MUST 补齐、降级为 internal-only，或删除该 public entry point

#### Scenario: 删除 public CLI 不新增 wrapper
- **WHEN** implementation 删除或降级一个 `kd-sensing-*` console script
- **THEN** 项目 MUST 不新增同名 alias、compat wrapper、deprecation trampoline 或旧命令 fallback
- **AND** docs MUST 指向保留的 public CLI、owner module、local/manual 命令或普通 unknown-command 行为

### Requirement: Internal-only CLI 不得成为隐藏 public API
`src/kd_sensing/cli/*.py` 中包含 console-style `main()`、parser 或用户可运行 workflow 的模块 MUST 要么声明为 public console script 并补齐生命周期锚点，要么降级为 internal helper 或删除。Shared CLI helper MAY 保留，但 MUST 不提供独立用户 workflow。

#### Scenario: module-only runnable CLI 被处置
- **WHEN** `src/kd_sensing/cli/<name>.py` 提供独立 `main()` 或 console parser，但 `pyproject.toml` 未声明对应 console script
- **THEN** implementation MUST 声明 public CLI 并补齐生命周期锚点，或删除/降级该 wrapper
- **AND** current docs MUST 不推荐未声明的隐藏 `python -m kd_sensing.cli.<name>` 作为稳定入口

#### Scenario: shared CLI helper 可保留
- **WHEN** CLI 模块只提供 argparse helper、配置覆盖 helper、shared exit handling 或轻量 common function
- **THEN** 它 MAY 作为 internal helper 保留
- **AND** architecture checks MUST 不把它误判为 public runnable entrypoint

### Requirement: Public CLI glue 必须保持薄
Package CLI 文件 MUST 只负责参数解析、轻量路径/配置处理、调用 owner module 和返回 user-facing exit code。训练 loop、评估 loop、benchmark aggregation、dataset preparation、report building 或 paper table 生成主逻辑 MUST 位于 owner module，而不是 CLI glue。

#### Scenario: CLI 文件不复制 workflow
- **WHEN** implementation 修改 `src/kd_sensing/cli/` 下的 public CLI
- **THEN** CLI 文件 MUST 不新增大段训练、评估、dataset parsing、benchmark suite 或 report aggregation 主逻辑
- **AND** 若发现主逻辑在 CLI 文件中，implementation MUST 迁回 owner module 或删除该入口

#### Scenario: public help 保持无副作用
- **WHEN** 用户运行保留 public CLI 的 `--help`
- **THEN** 命令 MUST 不读取真实 `dataset/`、不加载 checkpoint、不启动训练、不写入 runtime outputs
- **AND** help smoke MUST 能在 `kd_mm_beam` 环境中完成
