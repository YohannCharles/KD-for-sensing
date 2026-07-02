# project-import-surface-consolidation Specification

## Purpose
定义 package import 面、低价值 facade、re-export、`__all__` 和内部 owner 路径的收敛规则，使源码边界保持轻量且不依赖兼容聚合层。

## Requirements

### Requirement: 审计确认的低价值源码表面必须收敛
项目 MUST 对已审计确认无当前调用方、无公开入口、无 registry、无 current 文档/OpenSpec 消费且仅由自身测试覆盖的源码表面执行删除或合并。删除 MUST 同步移除只服务该表面的测试、维护索引条目和 inventory current 分类；合并 MUST 不新增兼容 wrapper 或二级聚合层。

#### Scenario: 删除孤立诊断模块
- **WHEN** `communication_state_features` 或等价诊断 helper 只有自身测试引用，且不属于 CLI、配置、README、docs、OpenSpec current spec 或维护索引 current entry
- **THEN** 本 change MUST 删除该源码模块和只服务它的测试
- **AND** 架构边界检查 MUST 不再把该模块登记为当前诊断 surface

#### Scenario: 删除未接入模型原型
- **WHEN** LiDAR pillar encoder 或等价模型原型没有 registry、config、trainer、dataset、CLI 或 current docs 接入
- **THEN** 本 change MUST 删除该原型或将其移出当前源码支持面
- **AND** 当前 LiDAR BEV workflow MUST 保持可用且不要求该原型存在

#### Scenario: 合并重复 output registry helper
- **WHEN** 两个诊断 owner 提供等价的 `OutputRegistry` 或输出清单 helper
- **THEN** 本 change MUST 只保留一个 owner helper 或内联为局部函数
- **AND** 合并后 MUST 不新增长期通用 registry 抽象

#### Scenario: 删除未使用 dev 依赖
- **WHEN** dev extra 中的依赖没有源码、测试、docs、OpenSpec 或配置引用
- **THEN** 本 change MUST 从 `pyproject.toml` 删除该依赖
- **AND** 删除 MUST 不改变 runtime dependencies

#### Scenario: 源码删减不删除本地产物
- **WHEN** 本 change 删除源码、测试、配置或依赖声明
- **THEN** 实现 MUST 不删除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 历史权重
- **AND** 若用户另行要求删除本地产物，流程 MUST 使用 runtime cleanup manifest 或单独显式确认

### Requirement: Ponytail 二阶段源码表面瘦身
项目 MUST 将审计确认的过度工程表面按可验证 wave 收缩。候选项包括兼容 facade、legacy wrapper、单实现注册表、重复治理表、只服务已删表面的测试 helper、无收益样板 import 和可由现有标准库或既有依赖替代的默认依赖。每个候选项 MUST 被归类为删除、合并、保留并说明理由，且源码瘦身 MUST 不删除本地数据或运行产物。

#### Scenario: 删除默认重依赖
- **WHEN** 某个默认依赖只被当前源码用于标准图像读取、路径探测或其它可由已保留依赖覆盖的轻量任务
- **THEN** 本 change MUST 用更小的现有依赖或标准库替换该调用
- **AND** `pyproject.toml` MUST 不继续把该依赖列为默认 runtime 依赖

#### Scenario: 删除兼容 facade
- **WHEN** 某个 facade 只 re-export 已有 owner 模块符号，且 README、当前 docs、OpenSpec current specs、CLI、registry 和测试均可迁到 owner 路径
- **THEN** 本 change MAY 删除该 facade
- **AND** 内部源码 MUST 不新增对该 facade 的 import 来维持旧路径

#### Scenario: 折叠单实现扩展点
- **WHEN** 某个 registry、adapter 或策略接口只有一个 identity/no-op 实现且没有当前配置选择面
- **THEN** 本 change MAY 将其内联为默认路径或局部 helper
- **AND** 若未来出现第二个真实实现，项目 MUST 通过新的 OpenSpec change 重新引入窄扩展边界

#### Scenario: 样板 import 独立 wave 删除
- **WHEN** 项目 Python 版本契约已确认不低于 3.10 且代码不依赖 future annotations 的旧版本语义
- **THEN** 本 change MAY 批量删除 `from __future__ import annotations`
- **AND** 该机械修改 MUST 与行为修改分开验证或在最终说明中明确验证范围

### Requirement: 剩余低价值项目表面必须分类收敛
项目 MUST 对 ponytail 审计确认的剩余低价值表面建立候选分类，并按删除、合并、保留、归档或后续 change 处理。候选范围 MUST 至少覆盖重复实体 YAML、薄 re-export facade、退役 route guard、无调用 helper、一次性分析脚本、重复小工具和大型治理测试。分类 MUST 记录公开 surface 风险、当前调用方、替代 owner、验证命令和回滚方式。

#### Scenario: 候选删除项有证据
- **WHEN** 开发者准备删除源码、配置、脚本或测试
- **THEN** 候选项 MUST 被证明不属于当前 package CLI、registry、canonical config、README/docs current 入口、OpenSpec current requirement 或必要 focused test 输入
- **AND** 删除计划 MUST 指向替代 owner、recipe、文档位置或说明无需替代

#### Scenario: 候选保留项有理由
- **WHEN** 某个候选项因 public API、人工样例、diagnostics manifest 或外部迁移风险被保留
- **THEN** inventory 或实现说明 MUST 记录保留理由和未来删除触发条件
- **AND** 项目 MUST 不为了保留该候选项新增兼容 wrapper 或第二套治理表

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

### Requirement: 退役保活测试必须删除或迁移
只用于证明退役类、旧 alias、旧 facade 或已删除 helper 仍可直接导入/forward 的测试 MUST 删除或改写为当前 owner、registry unknown-name、canonical config 或 CLI 行为测试。

#### Scenario: 退役模型 direct-forward 测试不再保活
- **WHEN** 已从 registry 退役的整模型类不再属于当前公开 API
- **THEN** 测试 MUST 不再直接实例化该类来证明其 forward 仍可用
- **AND** 相关覆盖 MUST 迁到当前 `modular_sequence`、feature extractor、registry unknown-name 或 config load 行为

### Requirement: 兼容 facade 收缩后 owner 路径成为当前入口
项目 MAY 删除不再属于 current public surface 的兼容 facade、legacy wrapper 和 re-export 模块。删除前，当前内部源码、README、docs、OpenSpec、tests 和示例 MUST 改用真实 owner 模块、canonical registry 名称、配置路径或 package CLI；删除后不得新增等价 wrapper 恢复旧入口。

#### Scenario: 内部代码迁出 facade
- **WHEN** 内部源码仍通过兼容 facade 导入当前实现
- **THEN** 本 change MUST 将导入改为真实 owner 模块或 registry/config 构建路径
- **AND** 架构边界测试 MUST 拒绝该 facade 重新成为内部依赖

#### Scenario: 外部兼容路径作为 breaking change 删除
- **WHEN** 某个历史 import 路径未被当前 docs、CLI、registry 或配置声明为支持入口
- **THEN** 本 change MAY 删除该路径
- **AND** 变更说明 MUST 将其标记为 breaking change 并给出当前 owner 路径或当前入口类别

### Requirement: 无价值 re-export facade 必须退出当前导入面
包级或子包级 facade 如果只 re-export owner 模块符号、没有独立行为、没有 current CLI/registry/config 依赖，且内部代码可直接导入 owner 模块，则项目 MUST 删除该 facade 或把它收缩为极薄 public shim。内部源码、测试和文档 MUST 使用真实 owner 路径，不得继续通过 facade 维持旧 import。

#### Scenario: 内部代码迁出 objective metadata facade
- **WHEN** 内部代码需要 prediction objective metadata helper
- **THEN** 代码 MUST 直接从 `kd_sensing.engine.objectives.metadata` 或对应 objectives owner 导入
- **AND** `kd_sensing.engine.objective_metadata` MUST 不再作为内部 helper import source

#### Scenario: data 和 datasets lazy export 不再扩展
- **WHEN** 代码需要 `DeepSense6GDataset`、`MMWDataset`、`SyntheticSequenceDataset`、sample helper 或 target-shot helper
- **THEN** 代码 MUST 从具体 owner 模块导入
- **AND** `kd_sensing.data` 或 `kd_sensing.data.datasets` MUST 不新增 lazy re-export 来保留旧路径

#### Scenario: fusion 旧类名 alias 不再作为迁移层
- **WHEN** 用户或测试导入已退役 fusion 旧类名 alias
- **THEN** 系统 MAY 使用普通 `AttributeError` 或 unknown import 失败
- **AND** 项目 MUST 不通过 `_REMOVED_ALIASES` 或等价 facade 表继续承诺每个退役类名的专属迁移错误

### Requirement: 内部模块不得使用星号导入表达公共 API
当前源码内部模块 MUST 使用显式 import 表达依赖。`from <module> import *` 只允许在明确保留的兼容 facade 中使用；业务 owner、runner、manifest、artifact、plot、scenario 或 predictive 模块 MUST 显式导入实际使用符号。

#### Scenario: JEPA benchmark 模块显式导入 common 符号
- **WHEN** 开发者修改 `src/kd_sensing/diagnostics/jepa_benchmark_*.py`
- **THEN** 修改后的 owner 模块 MUST 不新增 `from kd_sensing.diagnostics.jepa_benchmark_common import *`
- **AND** 需要的常量、类型和 helper MUST 显式列在 import 语句中

#### Scenario: public facade 保持薄层
- **WHEN** 某个模块被明确保留为 public facade
- **THEN** facade MAY re-export 稳定 API
- **AND** facade MUST 不承载 suite-specific helper 实现或成为内部模块的依赖来源

### Requirement: 小型类型别名文件不得替代标准类型
项目 MUST 不为单个 `dict[str, Any]`、`PathLike` 或等价标准类型保留独立 package 文件。只有跨多个 owner 共享且能减少真实复杂度的类型定义 MAY 保留；否则代码 MUST 直接使用标准类型或在真实 owner 模块中定义局部类型。

#### Scenario: 删除 AnyConfig 别名文件
- **WHEN** `_typing.AnyConfig` 只等价于 `dict[str, Any]` 且只有少量调用方
- **THEN** 本 change MUST 将调用方改为标准类型注解
- **AND** `src/kd_sensing/_typing.py` MAY 被删除

### Requirement: 低价值 facade 和 thin wrapper 必须收缩
项目 MUST 删除或最小化只 re-export owner 符号、只转发 CLI `main`、只维护旧 import path 或只镜像 `__all__` 的低价值 facade。保留 facade 时，facade MUST 是当前明确公开 API，且 MUST 不导入重依赖、不注册默认组件、不承载旧 alias 和长期实现逻辑。

#### Scenario: 删除 3 行 CLI wrapper
- **WHEN** 某个包内 CLI 文件只导入另一个模块的 `main` 并导出 `__all__`
- **THEN** `pyproject.toml` 的 console script MUST 直接指向真实 owner `main`
- **AND** 删除 wrapper 后对应 `kd-sensing-* --help` MUST 继续可运行

#### Scenario: 内部代码不用 package facade
- **WHEN** 内部源码或测试需要某个实现符号
- **THEN** 它 MUST 从真实 owner module 导入
- **AND** 它 MUST 不通过 package-level re-export、旧 alias facade 或 aggregate module 维持旧路径

### Requirement: 内部 `__all__` 镜像不得成为维护负担
内部模块 MUST 不为了镜像所有可见符号而维护大型 `__all__` 表。`__all__` 只允许用于稳定 public facade、明确 plugin/export 边界或避免 wildcard import 的必要模块。

#### Scenario: 删除无用 `__all__`
- **WHEN** 某个模块没有 current docs 推荐 wildcard import，也不是稳定 public facade
- **THEN** 本 change MAY 删除该模块的 `__all__`
- **AND** 显式 import 调用方 MUST 继续工作

### Requirement: 退役整模型类不作为包结构保留对象
已由 `modular_sequence` 或当前 whole-model exception 替代、且从 registry 移除的旧整模型类 MUST 不再作为直接导入 public API 保活。仍被当前路径使用的 feature extractor 或子组件 MUST 保留在 owner module。

#### Scenario: 删除旧整模型类保留特征提取器
- **WHEN** 旧 strong/lightweight modality model 已退出 registry 但同文件 feature extractor 仍被当前模块化模型使用
- **THEN** 实现 MUST 删除退役整模型类和 alias
- **AND** 实现 MUST 保留 feature extractor 并保持当前 `modular_sequence` 构建可用

### Requirement: 低价值聚合面必须收敛
项目 MUST 避免新增或保留没有 current public 契约价值的 package-level barrel、兼容 facade、单用途包装模块或重复 helper 聚合。仓库内部实现 MUST 直接导入职责明确的 owner module；只有 README、current spec、pyproject console script 或明确 public import 契约登记的路径才可保留薄入口。收敛实现 MUST 不恢复旧脚本入口、退役研究线入口或绕过 `src/kd_sensing` 包结构的运行方式。

#### Scenario: 内部实现不依赖 package barrel
- **WHEN** 包内训练、评估、诊断、数据或测试代码需要使用具体 owner 功能
- **THEN** 代码 MUST 从职责明确的 owner module 导入
- **AND** 代码 MUST 不为了少写 import 路径而依赖 package `__init__.py` 的重依赖 re-export

#### Scenario: 单用途包装模块被合并或登记
- **WHEN** 某个源码模块只包装一个 owner 函数、类或常量，且没有 current public import 契约
- **THEN** 实现 MUST 将该包装合并回调用点或 owner module
- **AND** 若暂缓合并，inventory MUST 登记其保留原因、owner、删除条件和 focused validation

#### Scenario: 重复 helper 不形成新杂物间
- **WHEN** 多个模块出现语义一致的 CSV、JSON、float、slug 或 path helper
- **THEN** 实现 MUST 优先复用已有 owner helper 或建立领域窄 helper
- **AND** 实现 MUST 不把这些 helper 放入会扩大轻量导入面的跨领域大 `utils` 聚合

#### Scenario: 收敛不触碰本地产物
- **WHEN** 开发者实施聚合面收敛、包装删除或 helper 合并
- **THEN** 变更 MUST 不删除、移动、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或其它本地运行产物
- **AND** 变更 MUST 只影响源码、测试、文档、配置或 OpenSpec artifact

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

### Requirement: 默认组件导入不依赖兼容模块
默认组件导入流程 MUST 注册 canonical 内置组件，同时保持 registry 本身轻量可导入。默认组件导入 MUST 不通过 `scenario9.py`、`engine.builders`、`data.transforms` 或 `_legacy` 兼容模块完成。

#### Scenario: 导入默认 dataset 组件
- **WHEN** 构建流程调用默认组件导入函数后再构建 DeepSense6G dataset
- **THEN** 默认导入 MUST 加载场景中立 dataset 模块
- **AND** 系统 MUST 不导入 `kd_sensing.data.datasets.scenario9`

#### Scenario: registry 轻量导入
- **WHEN** 开发者执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 dataset、model、training、checkpoint 或兼容 facade 模块

### Requirement: Registry helper surface 必须最小化
组件注册表 MUST 只暴露构建、查询、注册和错误诊断所需 API。无当前调用方、无 CLI、无 docs current 消费、无测试必要性的自检 helper MUST 删除，而不是作为 public API 长期保留。

#### Scenario: 删除 registry self check
- **WHEN** `registry_self_check` 没有项目内调用方且 registry 行为已由 focused tests 覆盖
- **THEN** 本 change MUST 删除该 helper 和 `__all__` 导出
- **AND** component registry tests MUST 继续覆盖 build、unknown name、duplicate name 和 missing required parameter 错误

#### Scenario: 不新增替代 smoke helper
- **WHEN** registry self check 被删除
- **THEN** 项目 MUST 不新增等价的长期 smoke function 或 CLI
- **AND** 必要验证 MUST 留在 pytest focused tests 中

### Requirement: Registry 公开导出不得镜像非 API 细节
`__all__` MUST 只包含真实 public API。删除 helper、removed alias 或内部 registry 表时，`__all__` MUST 同步收缩；项目 MUST 不为了保持旧导出而保留空 wrapper。

#### Scenario: 删除导出后 import 失败
- **WHEN** 非推荐外部代码从 registry 模块导入已删除 helper
- **THEN** 导入 MAY 失败
- **AND** 当前构建流程和 focused tests MUST 不依赖该 helper

### Requirement: Registry 不保活退役整模型类
组件 registry MUST 只保留当前 canonical 构建所需的 dataset、model、encoder、projector、core、head、loss、metric 和 preprocessor 名称。已经退役且不再注册的整模型类和旧 alias MUST 不通过直接导入测试或 facade 继续作为 current API。

#### Scenario: 退役模型 registry 构建失败
- **WHEN** 用户通过 `MODELS.build()` 请求已退役的 strong/lightweight/teacher/student 旧整模型名称
- **THEN** registry MUST 拒绝该名称
- **AND** 错误 MUST 使用现有 unknown-name 或保留的 removed-name 风格列出当前可用名称

#### Scenario: 当前组件仍可注册和构建
- **WHEN** 构建流程调用默认组件导入后构建 `modular_sequence`、当前 encoder、current fusion whole-model exception 或当前 loss/metric
- **THEN** registry MUST 保持变更前的构建行为
- **AND** 删除退役类 MUST 不影响这些当前注册名

### Requirement: Registry helper 不新增自检抽象
Registry 的最小契约 MUST 由 focused tests 覆盖。项目 MUST 不保留或新增只包装测试逻辑的 registry self-check helper 作为 runtime API。

#### Scenario: 删除 registry self-check helper
- **WHEN** registry build、duplicate、unknown 和 missing parameter 行为已由 tests 覆盖
- **THEN** 本 change MUST 删除只服务这些检查的 runtime self-check helper
- **AND** 删除 MUST 不影响 registry 构建当前组件
