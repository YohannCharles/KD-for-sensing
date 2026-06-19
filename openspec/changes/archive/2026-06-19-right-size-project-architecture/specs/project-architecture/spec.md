## ADDED Requirements

### Requirement: 项目架构右尺寸化必须基于 owner 职责而非全局数量
项目 MUST 使用 owner 职责、公开 surface、导入边界、热点预算、复用关系和验证覆盖来判断模块是否应拆分、合并或保留。Python 文件数、function 数和 import 数 MUST 作为架构审计基线和趋势信号，但 MUST NOT 单独作为要求合并或拆分的硬性目标。

#### Scenario: 文件数较多但职责清晰
- **WHEN** 架构审计发现某个区域存在较多 Python 文件
- **THEN** 审计 MUST 先判断这些文件是否对应独立 owner、thin CLI、focused tests、轻量导入边界或公开兼容 facade
- **AND** 系统 MUST NOT 仅因为文件数量高就合并这些模块

#### Scenario: 文件较少但函数过长
- **WHEN** 架构审计发现单个 owner 中存在超预算 orchestration 函数、长初始化函数或混合 schema/write/runtime 职责的实现
- **THEN** 系统 MUST 将其登记为 hotspot、monitor 或 split-next
- **AND** 拆分方向 MUST 指向稳定职责边界，而不是按固定行数机械切割

### Requirement: 热点拆分必须保持公开行为和本地产物边界兼容
热点模块拆分 MUST 只改变内部模块组织，不得改变公开 CLI 名称、console scripts、public import owner、配置路径、数据 split 语义、beam label 语义、指标口径、manifest schema、run metadata、默认输出路径或本地产物边界。

#### Scenario: 拆分公开 workflow owner
- **WHEN** 开发者拆分 BeamBench、trainer、dataset、diagnostics、viewer manifest 或 benchmark owner
- **THEN** 包内公开 import、CLI 入口和 console script MUST 继续指向同一 public surface
- **AND** focused tests MUST 覆盖该 workflow 的关键 schema、summary、metadata 或 metric 输出

#### Scenario: 拆分不触碰本地产物
- **WHEN** 开发者实施热点拆分或运行对应验证
- **THEN** 变更 MUST NOT 删除、移动、重写或提交 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或历史本地产物
- **AND** 新增临时验证产物 MUST 位于测试临时目录或已忽略的本地产物路径

### Requirement: 同 owner 低价值 helper 可以合并但不得恢复兼容聚合层
项目 MUST 允许将同一 owner 下单调用点、只服务 re-export、无独立 public contract、无复用价值或仅为降低行数而产生的 helper 合并回清晰 owner 模块。合并 MUST 不新增旧入口、跨领域 `helpers.py`、兼容聚合层、仓库根运行方式或退役研究线入口。

#### Scenario: 合并内部 helper
- **WHEN** 开发者合并一个只被同一 owner 使用的内部 helper 文件
- **THEN** 调用方 MUST 继续使用 owner 的公开 import 或已登记窄模块
- **AND** 架构边界测试或治理索引 MUST 更新为合并后的模块布局

#### Scenario: 禁止用兼容 wrapper 降低迁移成本
- **WHEN** helper 文件被合并或删除
- **THEN** 系统 MUST NOT 新增只转发旧 helper 路径的兼容 wrapper
- **AND** 内部代码 MUST NOT 从公开 facade 回流导入 suite-specific helper

### Requirement: 大 owner 保留必须有 accepted rationale 和验证命令
项目 MAY 保留较大的 owner 模块，但该 owner MUST 在维护索引或 inventory 中登记 `right-size-accepted` 或等价状态、accepted rationale、保留职责、验证命令和未来拆分触发条件。没有 accepted rationale 的超预算 owner MUST 被登记为 `split-next`、`monitor` 或 `defer-with-rationale`。

#### Scenario: 审计型 diagnostics owner 保留
- **WHEN** JEPA benchmark、visual analysis、run index 或 cleanup owner 因输出 schema 审计需要保持较大文件
- **THEN** 维护索引或 inventory MUST 记录该 owner 的职责边界、保留理由和 focused tests
- **AND** 新增实现 MUST NOT 回流到公开 facade 或轻量导入路径

#### Scenario: accepted owner 继续增长
- **WHEN** `right-size-accepted` owner 新增职责、超过既有 rationale 或触碰新的 public schema
- **THEN** 开发者 MUST 更新 accepted rationale 或将 owner 改为 split/monitor 状态
- **AND** 对应 focused tests MUST 覆盖新增职责

### Requirement: import 治理必须保护轻量导入边界
项目 MUST 将 import 治理重点放在 eager import、公开 facade 回流、跨领域依赖和重依赖泄漏上。轻量配置、路径、registry、package init、public facade 和 thin CLI MUST 不因架构整理额外导入 dataset reader、model implementation、training runtime、matplotlib、pandas、scipy、skimage、checkpoint 或权重文件。

#### Scenario: 轻量模块导入
- **WHEN** 开发者导入 `kd_sensing.config`、`kd_sensing.registries`、路径工具、包级公共 API 或已登记轻量 helper
- **THEN** 导入 MUST 成功且不触发训练、数据读取、模型权重加载或重型可视化依赖

#### Scenario: facade 内部回流
- **WHEN** 内部源码新增对公开 facade 中已迁移 helper 的 import 或调用
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向对应窄模块或 owner 模块作为迁移路径
