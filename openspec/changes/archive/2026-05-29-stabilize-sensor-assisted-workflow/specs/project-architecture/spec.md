## ADDED Requirements

### Requirement: MMW 入口生命周期 inventory 必须同步
新增或保留的 MMW Python 脚本、shell orchestration 和研究支持入口 MUST 具有可审计生命周期。项目表面积 inventory 与架构边界测试 allowlist MUST 同步记录入口类别、保留原因、推荐入口关系、输出产物边界和删除或收敛条件。

#### Scenario: 新增 MMW 脚本入口需要 inventory
- **WHEN** 开发者新增 `scripts/`、`scripts/mmw/`、`tools/analysis/` 或 `tools/visualization/` 下的 MMW Python 或 shell 入口
- **THEN** 架构边界检查 MUST 要求该入口出现在项目表面积 inventory 或等价生命周期文档中
- **AND** inventory MUST 说明该入口属于包内 CLI、薄 alias、研究诊断脚本、数据准备脚本或 shell orchestration 中的哪一类
- **AND** 对应测试 allowlist MUST 与 inventory 保持一致

#### Scenario: 未登记入口导致表面积检查失败
- **WHEN** 工作区中存在未登记的 MMW Python 或 shell 入口
- **THEN** 表面积回归检查 MUST 失败
- **AND** 失败信息 MUST 列出缺失登记的相对路径
- **AND** 失败信息 MUST 指向更新 inventory、删除重复入口或改为包内 CLI 的修复路径

#### Scenario: 重复 MMW orchestration 不成为推荐入口
- **WHEN** 多个 shell orchestration 覆盖同一 MMW quick validation 工作流
- **THEN** inventory MUST 标记推荐入口和补充 profile 的关系
- **AND** README 或 docs MUST 不把重复 shell wrapper 描述为唯一 canonical 入口
- **AND** 若已有包内 CLI 覆盖同一工作流，重复 shell wrapper MUST 标记为短期薄 alias 或研究脚本

### Requirement: HiST-Beam LOSO executor 热点拆分边界
HiST-Beam LOSO executor 继续增长时 MUST 按职责拆分到窄模块。公开 facade 可以保留现有 CLI 和 import 行为，但新增 preflight、stage orchestration、summary/conclusion 和 matrix metadata 逻辑 MUST 优先进入职责明确的内部模块。

#### Scenario: 新 preflight 逻辑进入窄模块
- **WHEN** 开发者新增或修改 MMW 数据可用性检查、prepared artifact 校验或 split materialization 检查
- **THEN** 主要实现 MUST 位于 preflight 或数据准备 adapter 模块
- **AND** executor facade MUST 只负责调用该模块并保持公开入口兼容

#### Scenario: 新 summary 逻辑不写入 stage 执行主体
- **WHEN** 开发者新增 quick validation conclusion、eligibility 汇总或 matrix metadata 写出逻辑
- **THEN** 主要实现 MUST 位于 summary/conclusion 或 matrix metadata 模块
- **AND** stage execution 模块 MUST 不承担最终结论排序和主结论 eligibility 解释职责

#### Scenario: 拆分后产物兼容
- **WHEN** executor 内部模块被拆分
- **THEN** 现有 run metadata、summary JSON、quick validation conclusion、checkpoint reuse metadata 和公开 CLI 参数 MUST 保持兼容
- **AND** focused characterization tests MUST 覆盖关键公开字段
