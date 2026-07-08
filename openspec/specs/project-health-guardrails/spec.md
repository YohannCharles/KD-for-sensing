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

### Requirement: Console script surface guardrail
项目健康护栏 MUST 检查 `pyproject.toml` console scripts、public CLI lifecycle 清单、CLI help smoke、project surface inventory 和 current docs/OpenSpec 的一致性。新增、删除或降级 public CLI 时，guardrail MUST 能发现缺少 lifecycle 分类、缺少 smoke、docs stale 引用或已删除命令回流。

#### Scenario: pyproject 与 help smoke 一致
- **WHEN** 开发者运行 CLI/architecture 健康检查
- **THEN** 检查 MUST 比对 `pyproject.toml` 中的 `kd-sensing-*` entry points 和保留 public CLI 的 help smoke 覆盖
- **AND** 缺少 smoke 的 public CLI MUST 被报告，除非 OpenSpec 明确将其标记为不需要 help smoke 的例外

#### Scenario: docs 不引用已删除 public CLI
- **WHEN** public console script 被删除或降级为 internal-only
- **THEN** README、docs、OpenSpec current specs 和 tests MUST 不再把旧命令描述为 current public entrypoint
- **AND** `kd-sensing-project-surface-doctor --scope cli-surface` MUST 能发现 current 文档中的 stale command reference

#### Scenario: 新 public CLI 需要生命周期锚点
- **WHEN** 后续 change 新增 `kd-sensing-*` console script
- **THEN** architecture/surface 检查 MUST 要求同步 owner module、inventory/docs 引用、help smoke 和输出边界
- **AND** 缺少这些锚点时检查 MUST 失败或在 doctor 中报告 error

### Requirement: 大规模表面清理必须有快速验收
项目 MUST 为大规模表面清理提供快速验收命令，覆盖 OpenSpec 校验、架构边界、CLI help 和被修改入口的引用一致性。所有项目相关 Python 验收 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 清理实现后的快速验收
- **WHEN** 支持面清理实现完成
- **THEN** 开发者 MUST 运行 `openspec validate project-health-guardrails --strict`
- **AND** 开发者 MUST 运行 `openspec validate --all --strict`
- **AND** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`

#### Scenario: 修改 CLI 或脚本入口后验收
- **WHEN** 清理实现修改 console script、local/manual runner 或可视化入口
- **THEN** 开发者 MUST 运行对应 `--help` 或无副作用 smoke 检查
- **AND** 检查 MUST 不读取真实 dataset、不启动训练、不写入新的源码内产物

#### Scenario: 公共入口瘦身验收
- **WHEN** implementation 修改 pyproject、package CLI、CLI module、inventory 或 current docs 中的 public entrypoint surface
- **THEN** 开发者 MUST 运行 `openspec validate --all --strict`
- **AND** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q`
- **AND** project surface doctor 的 CLI surface、scripts/configs/hotspots 或等价 scope MUST 能报告 stale public command reference、缺少 smoke 或生命周期锚点漂移

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
- **THEN** 开发者 MUST 运行 `openspec validate project-health-guardrails --strict`、`openspec validate --all --strict` 和 `conda run -n kd_mm_beam pytest -q`
- **AND** 若全量 pytest 因环境或本地数据缺失无法完成，最终说明 MUST 列出替代 focused 验证

### Requirement: Guardrails reject mixed runtime artifacts
本 change 的健康护栏 MUST 继续拒绝 tracked runtime artifacts，并 MUST 允许 ignored cache 噪声不影响常规测试。实施不得把 `dataset/` 真实数据、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event、`.pytest_cache` 或 `__pycache__` 纳入源码变更。

#### Scenario: Tracked artifact failure
- **WHEN** git tracked files 包含 `__pycache__`、`.pyc`、`.pytest_cache`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或非允许权重文件
- **THEN** architecture boundary 或 surface guard MUST 失败
- **AND** 未跟踪/ignored 的同类本地产物 MUST 不驱动常规架构边界测试失败

### Requirement: Guardrails validate pruned surface from source of truth
架构边界测试 MUST 从 `pyproject.toml`、真实 tracked paths、README/current docs、inventory lifecycle 和集中 retired-route guard 推导检查，不得维护完整脚本 allowlist、完整 tombstone 文件清单或完整 config 数据库镜像。

#### Scenario: Scripts checked from inventory or manifest
- **WHEN** tracked scripts 存在
- **THEN** 架构边界测试 MUST 验证每个脚本有 lifecycle 记录或由 retained generator/manifest 推导
- **AND** 测试 MUST 不复制完整脚本文案说明

#### Scenario: Retired route checked centrally
- **WHEN** retired route token 出现在 docs/specs/source/tests 中
- **THEN** 测试 MUST 验证其语境是 retired、historical、guard 或 migration
- **AND** 测试 MUST 不要求每条 retired route 拥有独立 tombstone spec

### Requirement: Validation covers deletion batches
每个删除批次 MUST 至少运行对应 focused validation：入口删除运行 CLI/architecture checks，config 删除运行 config characterization/generator checks，tombstone 折叠运行 OpenSpec strict 和 retired-route guard tests。

#### Scenario: Deletion batch validation
- **WHEN** 本 change 删除 CLI、script、config、spec 或 test 文件
- **THEN** 最终说明 MUST 记录已运行的 focused validation
- **AND** 未运行的验证 MUST 说明原因和剩余风险

### Requirement: 大测试拆分后健康护栏覆盖不得下降
Architecture 和 focused tests MAY 拆分为更小文件，但 MUST 继续覆盖 retired route 回流、tracked runtime artifact、current path/config 引用、facade 回流和 script lifecycle 检查。

#### Scenario: architecture boundary 拆分后仍拒绝关键回归
- **WHEN** tests are reorganized
- **THEN** `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` or its documented replacement MUST still fail on old entrypoint回流, tracked output artifacts and invalid current references
- **AND** documentation MUST point to the replacement command if the file is split

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

### Requirement: 架构边界检查的脚本分类红点必须通过 inventory 修复
项目健康护栏 MUST 继续拒绝未分类 tracked `scripts/` Python 或 shell 文件。发现未分类脚本时，修复 MUST 更新 project surface inventory、删除脚本或迁移为正式 owner 入口，不得通过放宽测试或新增重复 allowlist 掩盖漂移。

#### Scenario: 未分类脚本失败
- **WHEN** `scripts/` 下存在 tracked `.py` 或 `.sh` 文件
- **THEN** 架构边界测试 MUST 能发现未在 project surface inventory 或等价 current 文档中登记的脚本
- **AND** 失败信息 MUST 指向缺失登记的相对路径

#### Scenario: 登记后检查恢复
- **WHEN** 未分类脚本被登记为 research diagnostic、dataset preparation、config generator、figure helper 或 local/manual helper
- **THEN** 架构边界测试 MUST 在不读取真实 `dataset/`、不启动训练、不写入 runtime artifacts 的情况下通过该分类检查

### Requirement: Shared runtime profile routing focused test
项目健康护栏 MUST 覆盖 shared runtime 的单模态 input profile routing。新增或修改 `prepare_task_inputs`、`prepare_fusion_inputs` 或单模态 input preparation helper 时，focused tests MUST 验证 profile key 与 modality 一致，并且测试 MUST 不读取真实 `dataset/`、不启动训练、不写入 checkpoint。

#### Scenario: 单模态 profile 路由回归被测试发现
- **WHEN** 开发者运行 runtime profile routing focused test
- **THEN** 测试 MUST 覆盖 radar、gps 和 lidar 的 profile 透传
- **AND** 如果任一单模态任务读取其它 modality 的 profile，测试 MUST 失败并指出任务名和错误 profile key

#### Scenario: runtime 改动后的最小验证
- **WHEN** 变更触碰 shared runtime input preparation
- **THEN** tasks 或最终说明 MUST 至少列出对应 runtime focused test
- **AND** Python 验证命令 MUST 使用 `conda run -n kd_mm_beam`

### Requirement: 可复制 verify 入口
项目 MUST 提供或记录一个可复制的最小 verify 入口，用于聚合 OpenSpec strict、架构边界、CLI help、配置 characterization 和无数据 synthetic smoke。该入口 MUST 不启动真实训练、不读取真实 `dataset/`、不加载真实 checkpoint、不写入训练产物。

#### Scenario: 运行 quick verify
- **WHEN** 开发者运行项目记录的 quick verify 命令
- **THEN** 命令 MUST 覆盖 `openspec validate --all --strict`
- **AND** 命令 MUST 覆盖 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **AND** 命令 MUST 不读取真实本地数据或写入 checkpoint

#### Scenario: CLI/config verify 分层
- **WHEN** 变更触碰 console script、CLI parser、config loader 或 virtual config
- **THEN** 项目 MUST 记录可复制的 CLI/config verify 命令
- **AND** 该命令 MUST 使用 `conda run -n kd_mm_beam`

### Requirement: 最小环境声明
项目 MUST 提供 tracked 的最小环境声明或环境生成说明，用于重建无数据 smoke 验证环境。环境声明 MUST 区分 smoke/dev 依赖与 GPU 训练依赖，并 MUST 不包含本地数据路径、密码、token 或 checkpoint。

#### Scenario: 新机器重建 smoke 环境
- **WHEN** 维护者在新机器上准备运行无数据健康检查
- **THEN** 文档或环境文件 MUST 给出安装项目和运行 quick verify 的最小步骤
- **AND** 步骤 MUST 不要求真实 `dataset/`、`outputs/` 或 `All_models/` 可用

### Requirement: 轻量 lint 和脚本编译检查
项目 MUST 提供轻量 lint 或 compile 检查，用于在全量 pytest 前发现 Python 语法错误、脚本入口错误和明显文档引用漂移。该检查 MUST 不替代 focused tests。

#### Scenario: 脚本语法错误快速暴露
- **WHEN** tracked `scripts/` 或 package CLI 文件存在 Python 语法错误
- **THEN** 轻量检查 MUST 在真实训练前失败
- **AND** 失败信息 MUST 指向具体文件

### Requirement: 项目表面积 doctor
项目 MUST 提供只读表面积 doctor，用于检查 scripts、configs、hotspots 和文档引用的高风险漂移。Doctor MUST 不删除、不移动、不重写源码、配置、本地数据、输出、日志、cache 或 checkpoint。

#### Scenario: Doctor 只读运行
- **WHEN** 开发者运行项目表面积 doctor
- **THEN** doctor MUST 只读取 tracked 源码、配置、文档、OpenSpec 和必要的 git 文件清单
- **AND** doctor MUST 不修改 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或配置文件

#### Scenario: Doctor 输出可定位问题
- **WHEN** doctor 发现未分类脚本、失效 config 引用或热点超出登记边界
- **THEN** 输出 MUST 包含文件路径、问题类型、引用的权威来源和建议验证命令

### Requirement: Doctor 可纳入 quick verify
高风险表面积 doctor MUST 可作为 quick verify 的一部分运行，或至少在文档中记录为非平凡脚本/config/hotspot 改动前的推荐检查。

#### Scenario: 入口改动前运行 doctor
- **WHEN** 变更新增或修改 `scripts/`、`tools/analysis/`、`configs/` 或热点 owner
- **THEN** tasks MUST 列出对应 doctor 命令
- **AND** Python 命令 MUST 使用 `conda run -n kd_mm_beam`

### Requirement: Surface doctor 默认输出必须 issue-only
Project surface doctor 的默认输出 MUST 以问题、摘要和可执行 next action 为主。完整 pass inventory、allowlist dump、machine-readable governance table 或大段无问题清单 MUST 通过显式 opt-in flag 请求。

#### Scenario: 无问题时输出短摘要
- **WHEN** project surface doctor 运行且没有发现问题
- **THEN** 默认输出 MUST 是短摘要，说明检查 scope 和无问题状态
- **AND** MUST 不默认打印完整 pass inventory JSON 或大段逐项清单

#### Scenario: 完整清单显式 opt-in
- **WHEN** 协作者需要完整 inventory、allowlist 或 machine-readable dump
- **THEN** MUST 使用 `--dump-inventory` 或等价显式 flag
- **AND** 输出格式 MUST 在 help text 中说明适合审计/机器处理，而不是默认人读路径

### Requirement: Surface doctor 瘦身不得降低失败可诊断性
默认输出变短后，任何失败或 warning MUST 仍包含 scope、问题路径、原因和建议 next action。

#### Scenario: 有问题时保留 actionable detail
- **WHEN** surface doctor 检查发现 missing、stale、orphan、complete-unarchived 或 wrapper 回流问题
- **THEN** 默认输出 MUST 包含足够定位和修复的信息
- **AND** full dump flag MAY 提供额外上下文但不能成为理解失败的必要条件

### Requirement: Post-C2 guardrail 必须检查保护范围
项目健康护栏 MUST 在 post-C2 表面积清理中检查 protected inventory，防止 MMW/CSI、主线 YAML/manifest、final C2/U-MaskBeamJEPA 主线和 U-Mask fusion 分支被误删或被文档降级为 retired。

#### Scenario: protected path 缺失被发现
- **WHEN** 开发者运行架构边界测试或 project surface doctor 的清理验收 scope
- **THEN** 检查 MUST 验证 protected inventory 中的 MMW/CSI、主线 YAML/manifest、final C2/U-Mask owner 和 U-Mask fusion branch owner 仍存在或有明确替代记录
- **AND** 缺失且无替代记录 MUST 报告 error

#### Scenario: protected docs 不被标成 retired
- **WHEN** README、docs 或 OpenSpec current specs 描述 MMW、final C2、U-MaskBeamJEPA 或 protected mainline config
- **THEN** 健康检查 MUST 不允许这些 protected surface 被描述为 retired、historical-only 或 delete-candidate
- **AND** 若文档只描述后续审计候选，MUST 明确其不属于本 change 删除范围

### Requirement: Stale reference 检查必须覆盖删除波次
Post-C2 清理后，健康护栏 MUST 检查 current README、docs、OpenSpec specs、tests、pyproject 和 scripts 默认路径中是否仍引用已删除入口、已删除 config 或已删除 module。历史 archive 中的引用 MAY 保留，但 MUST 不被 current docs 当作推荐入口。

#### Scenario: 删除 CLI 后 docs stale reference
- **WHEN** public console script 被删除
- **THEN** `kd-sensing-project-surface-doctor --scope cli-surface` 或等价检查 MUST 报告 current docs/specs 中的 stale command reference
- **AND** CLI help smoke MUST 不再要求已删除命令存在

#### Scenario: 删除 config 后 current reference
- **WHEN** YAML 或 manifest 被删除或生成化
- **THEN** 配置引用检查 MUST 能发现 current docs、tests、scripts 或 specs 中指向不存在路径的引用
- **AND** 修复路径 MUST 是恢复配置、更新到 generator/manifest/base config，或把引用改为 historical

### Requirement: 清理验收必须保持无副作用
Post-C2 清理验收 MUST 只读取 tracked source、configs、docs、OpenSpec、tests 和 git metadata，不得读取真实 `dataset/`、启动训练、加载 checkpoint 或写入 `outputs/`、`logs/`、cache。

#### Scenario: 快速验收命令
- **WHEN** implementation 完成一个删除 wave
- **THEN** 验收 MUST 至少考虑 `openspec validate --all --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **AND** 触碰 CLI/config/script 时 MUST 追加 `make verify-cli-config`、`make verify-compile` 或对应 focused tests

#### Scenario: MMW 周边触碰追加 focused tests
- **WHEN** implementation 虽然保留 MMW 但修改了 MMW docs、configs、CLI lifecycle 或 guardrail
- **THEN** tasks 或最终说明 MUST 追加 MMW/CSI focused validation
- **AND** 验证 MUST 不要求真实 MMW dataset 内容进入源码变更
