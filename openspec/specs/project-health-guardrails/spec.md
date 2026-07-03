# project-health-guardrails Specification

## Purpose
定义项目健康护栏的检查层级、维护性热点 inventory、共享 pytest bootstrap、配置生命周期扫描和 OpenSpec 文档质量规则，使日常改动能快速发现入口漂移、旧路线回流、占位规范和本地产物边界问题。
## Requirements
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

### Requirement: 模型架构扩展护栏
项目健康护栏 MUST 检查新增模型、baseline 配置和扩展文档是否遵循模型架构扩展契约。检查 MUST 只读取已跟踪源码、配置、文档、OpenSpec artifact 和测试，不得读取真实 `dataset/`、`outputs/`、checkpoint 或 cache。

#### Scenario: 未说明例外的新整模型注册被发现
- **WHEN** 已跟踪源码新增 `@MODELS.register(...)` 或等价整模型注册名
- **THEN** 架构边界测试 MUST 要求该注册名出现在当前 specs、active change artifact、inventory 或明确 allowlist 中
- **AND** 缺少 whole-model exception 说明时检查 MUST 失败

#### Scenario: 文档默认示例绕过 modular_sequence 被发现
- **WHEN** 扩展指南把直接整模型注册描述为普通 baseline 的首选路径
- **THEN** 文档健康检查 MUST 失败
- **AND** 失败信息 MUST 要求改为 `modular_sequence` 或子组件注册默认示例

### Requirement: 新模型 metadata 护栏
项目健康护栏或 focused tests MUST 验证新增可训练模型和模块化子组件的 metadata 可审计。整模型例外 MUST 提供 `training_strategy_metadata()` 或等价 helper；模块化组件 metadata MUST 能被 `ModularSequenceModel` 聚合。

#### Scenario: 整模型例外缺少 metadata
- **WHEN** 新增 whole-model exception 被 registry 构建
- **THEN** focused test MUST 验证其 metadata 至少包含模型类型、启用模态、架构类别和 reliability metadata 消费状态
- **AND** 缺失关键字段时测试 MUST 失败

#### Scenario: 模块化组件 metadata 可聚合
- **WHEN** 新增 encoder、core 或 head 提供训练策略 metadata
- **THEN** focused test MUST 验证 `ModularSequenceModel.training_strategy_metadata()` 包含该组件信息
- **AND** run metadata helper MUST 不需要为每个组件新增专用分支

### Requirement: Batch 和 runtime 分支回流检查
项目健康护栏 MUST 防止新增 baseline 通过复制 batch preparation、forward task routing 或 validation loop 来绕开共享 runtime。模型改动如需新增输入契约，MUST 更新中心化模态或 batch metadata contract，并补充 focused tests。

#### Scenario: 新 baseline 不新增专用 forward 分支
- **WHEN** 新增普通 baseline 配置或组件
- **THEN** 实现 MUST 复用 `prepare_task_inputs`、`forward_task_model` 或现有共享 runtime
- **AND** 架构或 focused tests MUST 拒绝仅服务某个 baseline 的训练/验证 forward 分支回流

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

### Requirement: 项目健康检查可分层运行
项目 MUST 提供或记录一组快速健康检查，用于在不启动真实训练的情况下验证导入边界、CLI 入口和当前保留的核心诊断逻辑。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。健康检查 MUST 不再要求 Phase 1.5 或互补分析测试存在。

#### Scenario: 轻量导入 smoke
- **WHEN** 开发者运行项目健康检查中的轻量导入 smoke
- **THEN** 检查 MUST 验证配置、路径、模态契约、engine 轻量子模块和 diagnostics 轻量子模块可导入
- **AND** 检查 MUST 验证这些导入不触发指定重依赖模块

#### Scenario: 快速回归命令覆盖当前红点
- **WHEN** 开发者运行项目健康检查中的快速回归命令
- **THEN** 检查 MUST 覆盖架构导入边界、console script help 和当前仍保留的核心诊断逻辑
- **AND** 命令 MUST 能在全量 pytest 之前快速暴露项目结构回归

#### Scenario: 全量测试仍作为最终验收
- **WHEN** 变更实现完成
- **THEN** 开发者 MUST 使用 `conda run -n kd_mm_beam pytest -q` 作为最终回归验收
- **AND** 全量测试 MUST 通过

### Requirement: 架构增长回归检查
项目 MUST 提供快速架构回归检查，用于发现训练方法逻辑重新堆入 `trainer.py`、退役 viewer manifest/visualization 逻辑回流、或内部代码重新依赖二级兼容聚合层的问题。该检查 MUST 可在不启动真实训练的情况下运行，并 MUST 使用 `kd_mm_beam` 环境。检查 MUST 同时防止已退役的 G2D、CRAF、MARF、Multimodal-NF、viewer manifest、旧静态 visualization、GPS window、BGAM 和 DeepVerse/DT31 模块重新进入 active code path。

#### Scenario: 检查训练主循环扩张
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证新增训练方法主要通过扩展模块接入
- **AND** 测试 MUST 防止 `trainer.py` 新增退役 G2D、CRAF、MARF 等方法特有的大段私有 helper

#### Scenario: 检查 viewer manifest 回流
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证 viewer manifest CLI、helper、prediction export 和 tools wrapper 不存在
- **AND** 测试 MUST 防止旧 `kd_sensing.diagnostics.visualization` 包或仓库级 `tools/visualization` viewer support 回流

#### Scenario: 检查退役模块残留
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证 active import、registry 和配置推荐面不再引用 G2D、CRAF、MARF、Multimodal-NF、GPS window、DeepVerse/DT31、BGAM、viewer manifest 或旧静态 visualization
- **AND** 测试 MUST 不要求这些退役模块可导入

#### Scenario: 快速检查命令可运行
- **WHEN** 开发者执行项目记录的快速架构检查命令
- **THEN** 命令 MUST 在不读取真实数据集、不加载 checkpoint、不启动训练的情况下完成
- **AND** 命令 MUST 能在全量 pytest 前暴露架构边界回归

### Requirement: 项目表面积回归检查
项目 MUST 提供轻量表面积回归检查，用于发现源码变更中重新引入的本地产物、重复入口、已删除兼容路径和可生成配置实体化。该检查 MUST 不读取真实数据集、不加载 checkpoint、不启动训练，并 MUST 使用 `kd_mm_beam` 环境运行。

#### Scenario: 本地产物未进入源码表面积
- **WHEN** 开发者运行表面积回归检查
- **THEN** 检查 MUST 拒绝已跟踪的 `__pycache__`、`.pyc`、`.pytest_cache`、训练输出、日志、cache 和新生成 checkpoint
- **AND** 检查 MUST 允许 `dataset/.gitkeep` 这类明确的源码占位文件

#### Scenario: 重复入口回流被拒绝
- **WHEN** 项目中新增 `scripts/` 或 `tools/` Python 入口
- **THEN** 检查 MUST 判断该入口是否复制已有 `kd_sensing.cli.*` parser/main 或 console script 工作流
- **AND** 重复入口 MUST 被拒绝，除非对应 OpenSpec requirement 明确允许该 package CLI、研究脚本或数据准备脚本边界

#### Scenario: 表面积 inventory 可审计
- **WHEN** 开发者运行架构边界测试或专用 inventory 命令
- **THEN** 输出或测试断言 MUST 覆盖实体 YAML 数量、脚本入口数量、README/OpenSpec 待整理项和已知兼容入口 allowlist
- **AND** 新增 allowlist 项 MUST 通过 OpenSpec change 说明原因

### Requirement: 架构 guardrail 必须匹配真实支持面
架构边界测试、inventory 文档和当前支持入口 MUST 使用同一套项目表面定义。新增、迁移或删除配置、脚本和公开入口时，项目 MUST 同步更新 guardrail、inventory 和引用文档，不得通过过宽阈值掩盖真实漂移。

#### Scenario: 配置数量 guardrail 更新
- **WHEN** `configs/fusion/` 的当前支持 YAML 集合发生变化
- **THEN** 架构边界测试中的数量阈值或 allowlist MUST 与 inventory 中的分类一致
- **AND** 测试 MUST 继续限制根目录无限增长

#### Scenario: 脚本 allowlist 更新
- **WHEN** local/manual runner、thin CLI alias 或 research diagnostic 脚本引用的配置路径变化
- **THEN** 脚本、inventory 和测试 allowlist MUST 同步更新
- **AND** 当前脚本 MUST 不引用不存在的配置文件作为默认入口

#### Scenario: 脚本生命周期来自 inventory
- **WHEN** `scripts/` 下存在 tracked Python 或 shell 文件
- **THEN** 架构边界检查 MUST 从 project surface inventory、current docs 或 OpenSpec lifecycle 读取其分类事实
- **AND** 测试 MUST 不维护与 inventory 重复的完整脚本 allowlist，也不得通过放宽检查掩盖未分类脚本

### Requirement: 大规模表面清理必须有快速验收
项目 MUST 为大规模表面清理提供快速验收命令，覆盖 OpenSpec 校验、架构边界、CLI help 和被修改入口的引用一致性。所有项目相关 Python 验收 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 清理实现后的快速验收
- **WHEN** 支持面清理实现完成
- **THEN** 开发者 MUST 运行 `openspec validate cleanup-project-surface-drift --strict`
- **AND** 开发者 MUST 运行 `openspec validate --all --strict`
- **AND** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`

#### Scenario: 修改 CLI 或脚本入口后验收
- **WHEN** 清理实现修改 console script、local/manual runner 或可视化入口
- **THEN** 开发者 MUST 运行对应 `--help` 或无副作用 smoke 检查
- **AND** 检查 MUST 不读取真实 dataset、不启动训练、不写入新的源码内产物

### Requirement: 项目健康护栏纳入架构边界
项目 MUST 将健康护栏纳入架构边界测试，使包结构、轻量导入、入口 allowlist、热点 inventory、测试 bootstrap 和分层验证命令保持一致。架构边界测试 MUST 能在全量 pytest 之前暴露项目支持面或维护性边界漂移。

#### Scenario: 架构边界检查健康护栏文件
- **WHEN** 开发者运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **THEN** 测试 MUST 验证 shared pytest bootstrap、项目表面积 inventory 和当前健康检查命令说明存在且互相一致
- **AND** 测试 MUST 拒绝新增未分类的长期脚本入口、未登记热点或明显回流的兼容 facade helper

#### Scenario: 新增热点必须声明边界
- **WHEN** 新代码引入超长 orchestration 函数、dataset 类、manifest builder 或兼容 facade
- **THEN** 变更 MUST 同步更新热点 inventory 或拆分到职责明确的窄模块
- **AND** 架构边界测试 MUST 提供可定位到文件和符号的失败信息

### Requirement: 测试基础设施不得重复项目路径注入
项目 MUST 将普通测试的源码路径注入集中管理。除架构边界 import probe、subprocess smoke 或明确隔离环境测试外，测试文件不得各自维护重复的 `ROOT/SRC/sys.path.insert` 启动逻辑。

#### Scenario: 普通测试文件不复制 bootstrap
- **WHEN** 开发者新增或修改普通单元测试
- **THEN** 测试 MUST 通过 shared pytest bootstrap 导入 `kd_sensing`
- **AND** 文件级 `sys.path.insert` 复制片段 MUST 被架构边界测试拒绝或要求显式例外说明

#### Scenario: 子进程边界测试可控
- **WHEN** 架构边界测试需要在干净解释器中验证某个 import 不牵出重依赖
- **THEN** 该测试 MAY 在子进程代码中显式设置 `sys.path`
- **AND** 该例外 MUST 保持局部，不得作为普通测试模板传播

### Requirement: Baseline 与模型目录边界护栏
架构边界测试 MUST 验证 `src/kd_sensing/models/` 和 `src/kd_sensing/baselines/` 的依赖方向。`baselines/` workflow 模块 MUST 不注册 registry-backed 模型或模型子组件；`models/` 模块 MUST 不依赖 `kd_sensing.baselines` workflow package。轻量 package marker 和维护文档 MUST 与该边界一致。

#### Scenario: Baseline workflow 不注册模型组件
- **WHEN** 开发者在 `src/kd_sensing/baselines/` 下新增或修改 Python 文件
- **THEN** 架构边界测试 MUST 拒绝 `@MODELS.register`、`@ENCODERS.register`、`@PROJECTORS.register`、`@REPRESENTATION_CORES.register` 或 `@HEADS.register`
- **AND** 新模型能力 MUST 改放到 `src/kd_sensing/models/` 或通过配置复用已有组件

#### Scenario: 模型实现不反向依赖 workflow
- **WHEN** 开发者在 `src/kd_sensing/models/` 下新增或修改 Python 文件
- **THEN** 架构边界测试 MUST 拒绝从 `kd_sensing.baselines` 导入 workflow 实现
- **AND** 共享模型组件 MUST 通过 `models/`、`engine/`、`data/` 或其它真实 owner 模块复用

#### Scenario: 文档和 marker 不误导维护者
- **WHEN** 维护者查看 baseline package marker、模型目录或项目表面积 inventory
- **THEN** 文档 MUST 将 `baselines/` 描述为 workflow/paper reproduction owner
- **AND** 文档 MUST 不将 `baselines/` 描述为所有 baseline 模型的统一容器

### Requirement: Architecture boundary tests remain right-sized during streamlining
架构边界测试 MUST 在本 change 中继续验证结构事实和高风险回归，但不得复制完整源码目录清单、完整 OpenSpec prose、完整 scripts allowlist、完整 config 数据库或完整 hotspot budget 表。大型事实以 inventory、current specs、pyproject、真实 tracked paths 和 focused tests 为权威。

#### Scenario: 保留结构性失败
- **WHEN** current docs/specs 引用不存在的 config、console script、module path、public owner 或 lifecycle 分类
- **THEN** architecture boundary tests MUST 失败并指向修正文档、恢复文件或更新 lifecycle 分类
- **AND** 测试 MUST 不通过扩大阈值掩盖真实漂移

#### Scenario: 删除重复治理镜像
- **WHEN** 架构边界测试维护与 inventory、pyproject 或 OpenSpec 重复的大型 allowlist
- **THEN** 本 change MUST 删除该镜像或改为从权威来源直接推导
- **AND** 测试 MUST 仍覆盖旧入口回流、tracked runtime artifact、重依赖 barrel、facade 回流和 current path 引用失效

### Requirement: Streamlining waves have layered validation
每个 streamlining wave MUST 有分层验证：OpenSpec strict、architecture boundaries、目标领域 focused tests、公开 CLI/help smoke 或 import smoke。所有项目相关 Python 验证 MUST 使用 `conda run -n kd_mm_beam ...`。

#### Scenario: Wave focused validation
- **WHEN** wave 触碰 dataset、trainer/evaluation、model forward、diagnostics、config/scripts/import surface 或 docs/specs guardrail
- **THEN** tasks MUST 列出对应 focused test 命令
- **AND** wave 完成说明 MUST 记录实际运行结果、未运行原因和剩余风险

#### Scenario: Final regression
- **WHEN** 所有 waves 完成
- **THEN** 开发者 MUST 运行 `openspec validate streamline-project-architecture-waves --strict`、`openspec validate --all --strict` 和 `conda run -n kd_mm_beam pytest -q`
- **AND** 若全量 pytest 因环境或本地数据缺失无法完成，最终说明 MUST 列出替代 focused 验证

### Requirement: Guardrails reject mixed runtime artifacts
本 change 的健康护栏 MUST 继续拒绝 tracked runtime artifacts，并 MUST 允许 ignored cache 噪声不影响常规测试。实施不得把 `dataset/` 真实数据、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event、`.pytest_cache` 或 `__pycache__` 纳入源码变更。

#### Scenario: Tracked artifact failure
- **WHEN** git tracked files 包含 `__pycache__`、`.pyc`、`.pytest_cache`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或非允许权重文件
- **THEN** architecture boundary 或 surface guard MUST 失败
- **AND** 未跟踪/ignored 的同类本地产物 MUST 不驱动常规架构边界测试失败
