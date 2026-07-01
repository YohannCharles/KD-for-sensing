## ADDED Requirements

### Requirement: OpenSpec Purpose hygiene 必须被健康检查覆盖
项目健康护栏 MUST 扫描 current `openspec/specs/*/spec.md` 的 Purpose，拒绝 `TBD`、`created by archiving`、空 Purpose、未替换模板或其它归档脚手架文本。该检查 MUST 只读取已跟踪 OpenSpec artifact，不启动训练、不读取真实数据、不写入运行产物。

#### Scenario: current spec 保留归档 TBD
- **WHEN** current spec 的 Purpose 包含 `TBD - created by archiving`
- **THEN** 架构边界或 OpenSpec hygiene 检查 MUST 失败
- **AND** 失败信息 MUST 指向对应 spec 文件并要求补充真实 capability 边界说明

#### Scenario: OpenSpec validate 通过但 hygiene 失败
- **WHEN** `openspec validate --all --strict` 通过但项目自定义 hygiene 发现 scaffold Purpose
- **THEN** 项目 MUST 将其视为治理漂移
- **AND** 实施者 MUST 修复 Purpose 或在当前 change 中明确归档/折叠该 spec

### Requirement: 未分类脚本和配置必须被结构检查发现
项目健康护栏 MUST 验证 current 脚本、root fusion YAML、experiment YAML 和 root 文档的分类与真实文件系统一致。检查 MUST 优先读取 pyproject、真实路径、inventory 和 OpenSpec lifecycle，而不是维护重复的大型 allowlist。

#### Scenario: root fusion YAML 未登记
- **WHEN** `configs/fusion/*.yaml` 中存在 inventory 未分类的实体 YAML
- **THEN** 架构边界检查 MUST 失败
- **AND** 失败信息 MUST 要求迁移、删除或在 inventory 中登记其 root 保留理由

#### Scenario: 新脚本未登记
- **WHEN** `scripts/` 下新增 Python/shell 文件且不属于 ignored cache
- **THEN** 架构边界检查 MUST 要求 inventory 或 current docs 记录其 lifecycle、owner 和输出边界
- **AND** 未登记脚本 MUST 不通过测试静默进入 current surface

### Requirement: facade 回流检查必须区分内部源码和公开 CLI
项目健康护栏 MUST 检查内部源码是否从已登记 facade 导入窄 helper。检查 MUST 允许 facade 文件本身和 package CLI glue 使用公开 facade，但 MUST 拒绝 diagnostics、engine、data、models、losses、evaluation 和普通 tests 中新增 facade 回流。

#### Scenario: diagnostics 内部引用 benchmark facade
- **WHEN** diagnostics 内部模块从 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` 导入 benchmark 常量、schema helper 或 runner helper
- **THEN** 架构边界检查 MUST 失败
- **AND** 失败信息 MUST 指向 `jepa_benchmark_common.py`、`jepa_benchmark_manifest.py`、`jepa_benchmark_runner.py` 或对应 owner 模块

#### Scenario: CLI 兼容入口不误报
- **WHEN** `src/kd_sensing/cli/jepa_gps_shortcut_benchmark.py` 使用公开 benchmark facade
- **THEN** facade 回流检查 MUST 允许该引用
- **AND** 该允许范围 MUST 不扩展到内部 runtime 模块

### Requirement: 本地工具状态跟踪必须被拒绝
项目健康护栏 MUST 拒绝将本地工具运行状态纳入源码跟踪。检查 MUST 至少覆盖 `.codegraph/daemon.pid`、`.codegraph/*.sock`、`.codegraph/*.db*`、`.pytest_cache/`、`__pycache__/`、`.pyc`、`outputs/`、`logs/`、cache 和新生成 checkpoint。

#### Scenario: CodeGraph daemon pid 被 tracked
- **WHEN** `git ls-files` 返回 `.codegraph/daemon.pid`
- **THEN** 架构边界检查 MUST 失败
- **AND** 失败信息 MUST 说明该文件是本地 CodeGraph 运行状态，应移出 git 跟踪并由 `.codegraph/.gitignore` 覆盖

#### Scenario: ignored 本地状态存在但未 tracked
- **WHEN** 工作区存在 ignored CodeGraph pid/socket/db、pytest cache 或 Python bytecode
- **THEN** 常规架构检查 MUST 不因文件存在失败
- **AND** 检查 MUST 只拒绝这些路径被 git 跟踪

### Requirement: 普通 pytest bootstrap 不得重复
普通测试文件 MUST 依赖 shared pytest bootstrap、editable install 或 `tests/conftest.py` 提供的导入路径。除架构边界子进程 probe、隔离 import smoke 或明确局部测试 helper 外，测试文件 MUST NOT 在文件级复制 `ROOT/SRC/sys.path.insert` 启动片段。

#### Scenario: 普通测试复制 sys.path 启动片段
- **WHEN** 普通 `tests/test_*.py` 文件包含文件级 `sys.path.insert(0, str(SRC))`
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 要求改用 shared bootstrap 或将路径控制限制在隔离子进程 probe 中

#### Scenario: import probe 保留局部路径控制
- **WHEN** 架构边界测试在 subprocess code string 中显式设置 `sys.path`
- **THEN** 该用法 MAY 保留
- **AND** 它 MUST 不被抽成会 eager import runtime 模块的全局 helper
