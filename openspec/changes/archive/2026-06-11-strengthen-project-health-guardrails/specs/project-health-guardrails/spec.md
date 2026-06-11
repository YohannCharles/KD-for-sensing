## ADDED Requirements

### Requirement: 分层项目健康检查
项目 MUST 提供可记录、可重复的分层健康检查 workflow，用于在不启动真实训练、不读取真实数据、不写入源码内产物的前提下验证 OpenSpec、架构边界、CLI 入口和配置加载核心路径。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 快速健康检查覆盖架构和入口
- **WHEN** 开发者运行项目快速健康检查
- **THEN** 检查 MUST 至少覆盖 OpenSpec strict validate、架构边界测试、CLI help smoke 和配置加载 characterization
- **AND** Python 检查命令 MUST 使用 `conda run -n kd_mm_beam pytest ...`
- **AND** 检查 MUST 不启动真实训练、不读取 `dataset/` 真实数据、不写入 checkpoint 或训练输出

#### Scenario: 领域改动追加 focused tests
- **WHEN** 实现改动触碰训练、数据集、诊断、CLI、配置解析或模型 forward
- **THEN** tasks 或最终验证说明 MUST 列出对应 focused tests
- **AND** focused tests MUST 优先覆盖被修改 workflow 的公开契约，而不是只运行全量 pytest

### Requirement: 维护性热点 inventory
项目 MUST 维护一份可审计的维护性热点 inventory，记录已知超长模块、超长函数、超长类、兼容 facade、推荐窄模块拆分方向和暂缓原因。新增热点或热点显著扩大时，项目 MUST 更新 inventory 或拆分实现。

#### Scenario: 已知热点被命名
- **WHEN** 开发者审阅项目健康 inventory
- **THEN** inventory MUST 记录当前已知热点的文件路径、符号名、热点类型、当前规模指标和推荐拆分方向
- **AND** inventory MUST 包含训练主循环、DeepSense6G/MMW dataset、BeamBench Image AE+GPS workflow、run index、evaluation pass、batch preparation 和 manifest builder 等当前高维护成本区域

#### Scenario: 静态检查发现未登记热点
- **WHEN** 架构边界或健康检查发现新增超长函数、超长类或 facade 回流
- **THEN** 检查 MUST 失败或输出明确失败信息
- **AND** 失败信息 MUST 指向更新 inventory、拆分到窄模块或增加有理由例外这三种修复路径之一

### Requirement: 测试启动基础设施集中
项目 MUST 使用 shared pytest bootstrap 管理测试导入路径和通用 fixture。普通测试文件 MUST 不再复制 `ROOT/SRC/sys.path.insert` 启动片段；需要隔离 import 边界的子进程 probe MAY 显式控制 `sys.path`，但该例外 MUST 局限在对应 probe helper 内。

#### Scenario: 普通测试使用 shared bootstrap
- **WHEN** 新增普通测试文件需要导入 `kd_sensing`
- **THEN** 测试 MUST 依赖 shared pytest bootstrap 或 editable install
- **AND** 测试文件 MUST 不复制 `sys.path.insert(0, str(SRC))` 作为文件级启动逻辑

#### Scenario: import-boundary probe 保留显式路径控制
- **WHEN** 架构边界测试在子进程中验证轻量导入或重依赖隔离
- **THEN** probe helper MAY 在子进程代码中显式设置 `sys.path`
- **AND** 该路径控制 MUST 不被抽成会 eager import runtime 模块的全局 helper

### Requirement: 健康护栏不改变 runtime 语义
项目健康护栏 MUST 只检查源码、配置、文档、OpenSpec 和测试基础设施一致性，不得改变训练、评估、预处理、模型 forward、数据 split、beam label、checkpoint schema 或本地产物边界。

#### Scenario: 健康检查无副作用
- **WHEN** 用户运行健康检查命令
- **THEN** 命令 MUST 不删除、移动、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或真实本地运行产物
- **AND** 任何临时验证产物 MUST 位于 pytest 临时目录或 `.gitignore` 覆盖范围内

#### Scenario: 护栏实现不扩大公开入口
- **WHEN** 本 change 实现项目健康护栏
- **THEN** 系统 MUST 不新增长期训练/评估 CLI 或兼容 wrapper
- **AND** 若新增开发检查 helper，helper MUST 不成为 README 推荐的训练入口或旧研究路线替代入口
