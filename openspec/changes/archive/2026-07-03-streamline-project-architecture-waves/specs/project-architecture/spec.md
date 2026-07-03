## ADDED Requirements

### Requirement: Architecture streamlining campaign preserves public behavior
项目 MUST 允许按 wave 执行全仓架构收敛，但该收敛 MUST 只改变内部模块组织和未登记 public surface 的 import 路径，不得隐式改变当前 package CLI 名称、current canonical config 语义、dataset split 语义、beam label / label-space 口径、metric schema、checkpoint schema、run metadata 字段、默认输出分区或本地产物边界。

#### Scenario: 用户可见入口保持稳定
- **WHEN** architecture streamlining wave 修改 `data`、`engine`、`models`、`diagnostics`、`config`、`scripts` 或 `configs`
- **THEN** 当前 README、pyproject console scripts、current OpenSpec specs 和 inventory 登记的 package CLI MUST 继续可用
- **AND** 已登记 current workflow 的用户可见输入/输出契约 MUST 保持兼容

#### Scenario: 内部结构可以 breaking 收缩
- **WHEN** 某个 import path 未被 README、pyproject console script、current spec、inventory public surface 或 focused test 明确登记为 public owner
- **THEN** 该 path MAY 在本 change 中被删除、合并或迁到真实 owner
- **AND** 内部调用方 MUST 改为导入职责明确的 owner module，不得新增兼容 wrapper 维持旧路径

### Requirement: Architecture streamlining starts from a clean or documented baseline
项目 MUST 在实施任何源码 wave 前记录工作树、active change 和验证 baseline。若工作树存在无关实验改动、未跟踪配置/脚本、本地 cache 噪声或已完成但未归档的 active change，实施说明 MUST 先归档、提交、隔离，或明确记录 deferral 和影响范围。

#### Scenario: Wave 0 captures baseline state
- **WHEN** 本 change 进入 implementation
- **THEN** tasks 或实现说明 MUST 记录 `openspec list --json`、`git status --short`、已知未跟踪实验表面、产物边界占位文件状态和 baseline validation 命令结果
- **AND** 后续源码 wave MUST 不把无关实验变更或本地运行产物混入架构重构 diff

#### Scenario: 已完成 active change 不被误用
- **WHEN** active change 显示 status 为 complete
- **THEN** 本 change MUST 先归档该 change，或在 Wave 0 中说明暂不归档的原因、风险和与本 change 的隔离方式

