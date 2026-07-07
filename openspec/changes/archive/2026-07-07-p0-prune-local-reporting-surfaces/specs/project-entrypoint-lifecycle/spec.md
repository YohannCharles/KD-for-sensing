## ADDED Requirements

### Requirement: 本地报告脚本必须在结论沉淀后退出 current surface
项目 MUST 将一次性研究分析、论文表格排版、展示材料导出和局部结论脚本视为有生命周期的本地报告面。若其结论、输入来源和关键输出已经沉淀到 docs、claim notes、paper tables 或 retained artifact 说明，implementation MUST 删除脚本或合并到明确 owner，而不是继续把单次脚本保留为 current entrypoint。

#### Scenario: 一次性 analysis 脚本删除
- **WHEN** `scripts/analysis/` 中的脚本只复现已经沉淀的分析结论
- **THEN** 项目 MUST 删除该脚本或将其标记为非 current retained artifact source
- **AND** docs、OpenSpec current specs、tests 和 inventory MUST 不再要求该脚本路径存在

#### Scenario: 报告脚本合并不新增 wrapper
- **WHEN** 多个报告脚本只有输入 glob、标签或输出格式不同
- **THEN** 项目 MUST 收敛到一个 owner command 或 owner module helper，并通过显式参数表达差异
- **AND** 项目 MUST 不新增 alias、compat wrapper、deprecation trampoline 或同职责转发脚本

### Requirement: 删除本地报告脚本必须保留证据链
删除报告脚本前，implementation MUST 保留足够证据说明该脚本产生的结论仍可追溯。证据 MAY 是正式 docs、paper table、claim note、retained artifact manifest、或 canonical command 的输出契约。

#### Scenario: 删除前记录替代 owner
- **WHEN** 一个本地报告脚本从 current surface 删除
- **THEN** `docs/project_surface_inventory.md` 或相关 docs MUST 记录替代 owner、历史用途或 retained-with-reason
- **AND** 若输出支撑正式 claim，字段名、排序、筛选条件或 artifact 路径模式 MUST 可在替代 owner 中验证
