## ADDED Requirements

### Requirement: 健康护栏拒绝陈旧 OpenSpec validation 命令
项目健康护栏 MUST 验证当前维护文档和机器可读索引中的可复制 validation 命令仍指向存在的 OpenSpec spec、active change 或通用 strict 校验。已归档 change 的历史 validation 命令 MAY 出现在 archive artifact、历史记录或明确标注为 historical 的上下文中，但 MUST NOT 出现在当前 focused validation 列表中。

#### Scenario: 当前 focused 命令引用 missing change
- **WHEN** 架构边界测试扫描 `docs/maintainer_context_index.yaml`、AI 导航或 project surface inventory 中的当前 focused validation 命令
- **AND** 命令包含 `openspec validate <name> --strict`
- **AND** `<name>` 既不是 active change，也不是 current spec 或通用 OpenSpec 校验
- **THEN** 健康护栏 MUST 失败
- **AND** 失败信息 MUST 指向包含陈旧命令的文件和替代命令方向

### Requirement: 普通测试不得维护 tests 路径 bootstrap
普通 `tests/test_*.py` 文件 MUST 依赖 shared pytest bootstrap、仓库根路径或 package-style import 访问测试 helper。除架构边界 import probe、subprocess smoke 或显式隔离环境测试外，普通测试文件 MUST NOT 在文件级插入 `tests/` 目录到 `sys.path`。测试 helper MAY 继续放在 `tests/` 下，但调用方 MUST 使用 shared bootstrap 可解析的导入路径。

#### Scenario: 普通测试插入 tests 路径
- **WHEN** 普通 `tests/test_*.py` 文件包含文件级 `sys.path.insert(0, str(TESTS))` 或等价 `tests/` path 注入片段
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 要求改用 `from tests.<helper> import ...` 或 shared pytest bootstrap 支持的导入方式

#### Scenario: 子进程 import probe 保留显式路径
- **WHEN** 架构边界测试或隔离 import smoke 在 subprocess code string 中显式设置 `sys.path`
- **THEN** 该用法 MAY 保留
- **AND** 该例外 MUST 不被普通测试文件复制为 helper 导入模板

### Requirement: 架构边界检查 inventory 统计口径
架构边界测试 MUST 能发现 project surface inventory 中缺失统计口径说明的架构尺寸基线。检查 MUST 验证基线段至少说明扫描范围、统计来源或口径、排除的本地产物类别，以及这些数字不是硬拆分 KPI。

#### Scenario: inventory 数量没有口径说明
- **WHEN** `docs/project_surface_inventory.md` 更新源码、测试、脚本或 YAML 数量基线
- **THEN** 架构边界测试 MUST 验证该段包含统计口径、扫描范围和排除项说明
- **AND** 若只给出裸数字且没有用途/排除项，检查 MUST 失败

### Requirement: 可选性能噪声清理必须保持 focused 行为验证
健康护栏 MAY 包含不改变语义的性能或 warning 噪声清理任务，但这类任务 MUST 有 focused behavior validation，并 MUST 不通过 warning filter 掩盖真实数据契约问题。

#### Scenario: MMW DataFrame fragmentation warning 清理
- **WHEN** 实现修改 MMW helper 的 DataFrame 列构造方式以消除 fragmentation warning
- **THEN** 开发者 MUST 运行 MMW focused tests 验证 sample fields、metadata、label 和 preparation contract 不变
- **AND** 实现 MUST 不读取真实 `dataset/`、不写入 `outputs/`、不改变 sensor-assisted input/target 边界
