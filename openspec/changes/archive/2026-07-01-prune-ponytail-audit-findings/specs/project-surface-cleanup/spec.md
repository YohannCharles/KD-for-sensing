## ADDED Requirements

### Requirement: Ponytail 审计表面必须分类收口
项目 MUST 将 ponytail 审计确认的临时配置、一次性脚本、root runbook、薄 facade、重复 helper 和本地工具状态分类为删除、迁移、保留或后续 change。分类 MUST 记录 owner、当前调用方、公开 surface 风险、替代入口、验证命令和回滚方式。未分类项 MUST 不得作为 current README、docs、OpenSpec 或 package CLI 推荐入口。

#### Scenario: 新增脚本被分类
- **WHEN** `scripts/` 或 `tools/analysis/` 下存在新增 Python/shell 脚本
- **THEN** inventory MUST 将其分类为 package_cli、research_diagnostic、dataset_preparation、figure_helper、shell_orchestration 或 local/manual artifact
- **AND** 未分类脚本 MUST 不得被 README、docs 或 OpenSpec 描述为当前推荐入口

#### Scenario: 临时配置不进入 root canonical surface
- **WHEN** 新增配置只服务 Scene31/RBMA queue/fullrun/strong-encoder/seed sweep 或其它本地实验编排
- **THEN** 配置 MUST 位于语义明确的 experiment 子目录、被归档说明，或被删除
- **AND** 配置 MUST 不直接混入 `configs/fusion/*.yaml` 根目录，除非 inventory 将其登记为 canonical/current thin entry

### Requirement: 内部代码不得通过公开 facade 回流导入 owner helper
公开 facade MAY 保留外部兼容 import 或 CLI glue，但内部源码 MUST 直接导入职责明确的 owner 模块。新增内部引用不得从 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark`、`kd_sensing.data.mmw.preparation` 或其它已登记 facade 导入已迁出的窄 helper，除非该文件本身就是 facade 或 package CLI glue。

#### Scenario: JEPA visual analysis 直连 benchmark owner
- **WHEN** `kd_sensing.diagnostics.jepa_visual_analysis` 需要 benchmark suite 常量或 analysis bundle reader
- **THEN** 它 MUST 从 `jepa_benchmark_common.py`、`jepa_benchmark_runner.py` 或对应窄 owner 导入
- **AND** 它 MUST NOT 通过 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` facade 获取这些 helper

#### Scenario: CLI 仍可使用公开 facade
- **WHEN** package CLI 需要调用 GPS shortcut benchmark runner
- **THEN** CLI MAY 继续使用公开 facade 或直接 owner
- **AND** 该例外 MUST 不允许 diagnostics、engine、data、models 或 tests 内部新增 facade 回流

### Requirement: CodeGraph 本地运行状态不得作为源码产物
CodeGraph 索引、daemon pid、socket、WAL、cache、log 和 hook marker MUST 被视为本地运行状态。源码仓库 MAY 跟踪 `.codegraph/.gitignore` 这类忽略规则，但 MUST NOT 跟踪 `.codegraph/daemon.pid`、daemon socket、数据库、WAL、cache 或其它会随本机进程变化的文件。

#### Scenario: daemon pid 不被跟踪
- **WHEN** 开发者运行 `git ls-files .codegraph/daemon.pid`
- **THEN** 该路径 MUST 不在 git tracked 文件列表中
- **AND** `.codegraph/.gitignore` MUST 覆盖 pid、socket、数据库和 cache 等本地 CodeGraph 状态

#### Scenario: CodeGraph 仍可本地运行
- **WHEN** CodeGraph daemon 在本地生成 pid、socket 或数据库文件
- **THEN** 这些文件 MAY 保留在工作区供本地工具使用
- **AND** 架构边界检查 MUST 不要求提交这些文件

### Requirement: Fusion 根配置与 inventory 必须一致
`configs/fusion/` 根目录的实体 YAML MUST 与 `docs/project_surface_inventory.md` 中的 root canonical/current 分类一致。若 root YAML 集合变化，项目 MUST 同步更新 inventory、引用文档和架构边界测试；若文件不属于 root canonical/current thin entry，项目 MUST 将其迁入 experiment 子目录、归档或删除。

#### Scenario: root YAML 集合被验证
- **WHEN** 架构边界测试扫描 `configs/fusion/*.yaml`
- **THEN** 每个根 YAML MUST 出现在 inventory 的 root 保留分类中
- **AND** 未登记 root YAML MUST 被视为支持面漂移

#### Scenario: 实验 YAML 迁移后引用同步
- **WHEN** root fusion YAML 被迁移到 `configs/fusion/experiments/<family>/`
- **THEN** README、docs、scripts、tests 和 OpenSpec 中的 current 引用 MUST 指向新路径或移除
- **AND** 历史引用 MUST 明确标记为 historical、retired 或 local/manual
