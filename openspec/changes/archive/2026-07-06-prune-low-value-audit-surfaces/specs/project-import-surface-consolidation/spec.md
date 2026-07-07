## ADDED Requirements

### Requirement: 审计确认的小型 facade 和 writer 聚合必须收缩
项目 MUST 删除或合并本轮 ponytail-audit 已确认的低价值源码表面，包括只 re-export owner 符号的 difficulty package facade、只服务 U-mask Beam JEPA eval matrix 的 export writer 聚合、重复测试 path 样板，以及可由标准库直接表达的 JSON 往返 deep copy。收缩 MUST 不新增兼容 alias、二级聚合层或跨领域 `utils` 杂物间。

#### Scenario: difficulty package facade 不作为内部导入源
- **WHEN** 内部源码或测试需要 difficulty schema、profile、operator plan 或 pipeline helper
- **THEN** 代码 MUST 从 `kd_sensing.data.difficulty.schema`、`kd_sensing.data.difficulty.pipeline` 或真实 owner module 导入
- **AND** `kd_sensing.data.difficulty` package marker MUST 不继续作为内部 re-export barrel 承载这些符号

#### Scenario: eval export writer 不成为通用 eval 聚合面
- **WHEN** CLI、trainer runtime helper 或测试需要写出 U-mask Beam JEPA eval matrix 的 CSV、JSON 或 Markdown 结果
- **THEN** 代码 MUST 从 U-mask Beam JEPA eval matrix owner 或其窄 helper 调用 writer
- **AND** 项目 MUST 不保留 `kd_sensing.eval.export` 作为跨领域小型 helper 聚合模块

#### Scenario: JSON deep copy 使用标准库
- **WHEN** 代码需要复制默认 dict/list 配置并在本地 merge override
- **THEN** 实现 MUST 使用 `copy.deepcopy` 或等价标准库/已有 owner helper
- **AND** 实现 MUST 不用 `json.loads(json.dumps(...))` 作为普通深拷贝方式

#### Scenario: tests path 样板不重复维护
- **WHEN** `tests/conftest.py` 已统一把仓库根目录和 `src/` 加入 `sys.path`
- **THEN** 普通测试文件 MUST 不重复声明 `ROOT`、`SRC` 和本地 `sys.path.insert` 样板
- **AND** 删除样板后测试 MUST 继续通过同一 focused validation

### Requirement: Dataset descriptor 简化必须有净收益
`dataset_descriptors` MAY 从 dataclass wrapper 简化为静态 mapping 和 query functions，但只有在保持现有调用 API、验证错误语义、profile resolution 行为和 focused tests 的前提下，且实现更短更清楚时才可落地。若简化无法减少维护负担，implementation MUST 保留现状并记录 retained-with-reason。

#### Scenario: descriptor API 保持稳定
- **WHEN** data factory、config validation 或测试查询 dataset descriptor、profiles 或 metadata
- **THEN** `dataset_descriptor`、`list_dataset_descriptors`、`resolve_dataset_profiles` 和 `descriptor_metadata` 的行为 MUST 保持兼容
- **AND** 未知 dataset 或 profile 的错误 MUST 继续指向可操作的 supported values

#### Scenario: 无净收益则不改 descriptor 实现
- **WHEN** 静态 mapping 改写导致更多样板、较差错误信息或更模糊类型边界
- **THEN** implementation MUST 不强行删除 dataclass 层
- **AND** inventory 或实现说明 MUST 记录保留理由和未来可重新审计的触发条件

### Requirement: 本轮源码瘦身不得触碰本地产物
本 change 的删除、合并和测试样板清理 MUST 只影响源码、测试、文档、OpenSpec 或 inventory。实现 MUST 不删除、移动、压缩或重写本地数据、训练输出、日志、cache、checkpoint 或历史权重。

#### Scenario: prune change 不修改 runtime artifacts
- **WHEN** 开发者检查本 change 的 git diff
- **THEN** diff MUST 不包含 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 运行产物
- **AND** 若需要清理运行产物，MUST 使用单独显式请求或 runtime artifact cleanup 流程
