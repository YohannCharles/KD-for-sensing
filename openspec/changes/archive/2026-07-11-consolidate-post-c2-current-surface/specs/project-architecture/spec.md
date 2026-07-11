## ADDED Requirements

### Requirement: Current runtime 保持最小依赖流
当前运行架构 MUST 以 package CLI、config、data/difficulty、engine、models/losses 和 ignored runtime artifacts 构成最小依赖流。只服务 retired workflow、展示产品或治理镜像且没有 current consumer 的模块 MUST 不保留在 `src/kd_sensing`。

#### Scenario: 零消费者 owner 退出源码
- **WHEN** tracked import、config、CLI、script、current spec 和 claim provenance 审计均未发现某 owner 的 current consumer
- **THEN** implementation MUST 删除该 owner 和专属测试
- **AND** implementation MUST 不把 owner 合并进其它大文件来规避删除

#### Scenario: supporting owner 有真实下游
- **WHEN** run index 被 runtime cleanup 消费，或 JEPA mean pooling 被 current MMW config 消费
- **THEN** 对应 owner MUST 保留其最小 consumer contract
- **AND** 只服务 retired branch 的 sibling 实现 MUST 可独立删除

## MODIFIED Requirements

### Requirement: 安装入口与 pyproject 声明一致
项目 MUST 确保 editable install 后的 console scripts 与 `pyproject.toml` 的 `[project.scripts]` 声明一致。README 或工具文档中推荐的 package CLI MUST 可在 `kd_mm_beam` 环境中直接调用。保留的 console script MUST 是 parser/config glue，不得复制长期维护的 parser 或主实现。

#### Scenario: 退役和低价值 scripts 不声明
- **WHEN** 开发者检查 `pyproject.toml` entry points
- **THEN** 项目 MUST 不声明 viewer、research dashboard/preview、project surface doctor、training throughput、dataset audit 或其它已退役入口
- **AND** 项目 MUST 不通过 module alias 或 script wrapper 恢复这些命令

#### Scenario: 安装元数据刷新后入口齐全
- **WHEN** 开发者在 `kd_mm_beam` 中执行 `python -m pip install -e .`
- **THEN** entry points MUST 包含 train、evaluate、preprocess、runs、runtime cleanup/organize、paper export、U-Mask eval matrix、MMW GPS v2 和 MMW physics inspect 共十个 current 命令
- **AND** CLI help smoke MUST 从同一 pyproject current surface 验证这些入口

### Requirement: Architecture streamlining campaign preserves public behavior
项目 MUST 允许按 wave 删除明确登记的 public surface 和 internal surface。删除 MUST 在 change spec、pyproject、README、inventory 和 tests 中同步声明；未列入删除范围的 current canonical config、dataset split、beam label/label-space、metric schema、checkpoint schema、run metadata 和默认输出分区 MUST 保持兼容。

#### Scenario: 保留用户可见入口稳定
- **WHEN** architecture streamlining wave 修改 data、engine、models、diagnostics、config、scripts 或 configs
- **THEN** 本 change 明确保留的十个 package CLI 和 protected current workflow MUST 继续可用
- **AND** focused tests MUST 验证其用户可见输入输出契约

#### Scenario: 登记的 breaking surface 可删除
- **WHEN** command、module-only CLI、script path 或 internal import 已在 active change 中标记为 removed
- **THEN** implementation MAY 删除该 path 并同步所有 current references
- **AND** implementation MUST 不新增兼容 wrapper 维持旧路径

## REMOVED Requirements

### Requirement: 同 owner 低价值 helper 可以合并但不得恢复兼容聚合层
**Reason**: 与“内部 helper 合并与边界保留”重复，继续保留只增加规格镜像。
**Migration**: 统一使用“内部 helper 合并与边界保留”及本 change 的零消费者删除要求。

#### Scenario: 重复 helper 规则折叠
- **WHEN** 维护者判断 helper 应合并或删除
- **THEN** 只需遵守保留的 owner/helper requirement
- **AND** 不再维护第二份等价要求

### Requirement: Surface pruning preserves current user behavior
**Reason**: 与更新后的 “Architecture streamlining campaign preserves public behavior” 重复，且旧表述无法表达本 change 的显式 public breaking removals。
**Migration**: 使用更新后的 streamlining requirement 区分 retained behavior 与登记删除 surface。

#### Scenario: Surface pruning 规则归一
- **WHEN** implementation 删除 current 或 internal surface
- **THEN** 行为判断 MUST 依据更新后的 streamlining requirement
- **AND** 不再由重复 requirement 产生冲突
