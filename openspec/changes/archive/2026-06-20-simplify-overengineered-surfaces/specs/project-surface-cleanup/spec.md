## ADDED Requirements

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
