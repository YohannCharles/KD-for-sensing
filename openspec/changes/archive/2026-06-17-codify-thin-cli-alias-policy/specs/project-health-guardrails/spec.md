## ADDED Requirements

### Requirement: 薄 CLI alias 健康检查
项目健康护栏 SHALL 检查 CLI 和 scripts 入口不变厚。检查 MUST 基于 maintainer context index 中的 entrypoint lifecycle、owner module 和 output boundary，拒绝未登记入口和明显复制 workflow 逻辑的 thin alias。

#### Scenario: 新脚本缺少 owner module
- **WHEN** `scripts/`、`tools/analysis/` 或 package CLI 新增入口
- **THEN** 维护上下文索引 MUST 登记 owner module、responsibility 和 output boundary
- **AND** 缺少登记时架构边界测试 MUST 失败

#### Scenario: thin alias 包含训练循环 marker
- **WHEN** lifecycle 为 `thin_cli_alias` 的脚本包含大段训练循环、模型 forward、optimizer step 或 dataset parsing 主逻辑
- **THEN** 健康检查 MUST 失败或要求重新分类为 owner module
- **AND** 修复路径 MUST 是委托包内实现或创建正式 package module
