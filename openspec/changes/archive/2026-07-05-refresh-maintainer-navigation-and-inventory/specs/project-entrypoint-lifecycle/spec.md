## ADDED Requirements

### Requirement: Scene31/Scene31-34 报告脚本必须分类为本地研究报告表面
Scene31 和 Scene31-34 的论文表格、per-scene summary、final conclusion 等报告脚本 MAY 保留为 research diagnostic 或 local/manual reporting surface，但 MUST 在 inventory 或 current 文档中登记 lifecycle、职责和输出边界。它们 MUST 不作为 package CLI、README quickstart 或长期 public API 推荐入口。

#### Scenario: 报告脚本有输出边界
- **WHEN** 项目保留 Scene31 或 Scene31-34 报告脚本
- **THEN** inventory MUST 说明脚本读取本地 summary、fresh-eval 或 paper table 输入
- **AND** 输出边界 MUST 限定在 ignored `outputs/`、`logs/` 或显式用户路径，不得提交生成表格、结论、checkpoint 或 metrics

#### Scenario: 报告脚本不升级为 package CLI
- **WHEN** README、AGENTS、OpenSpec 或 docs 描述当前推荐入口
- **THEN** 这些报告脚本 MUST 不被描述为训练、评估、预处理或诊断 package CLI 的替代入口
- **AND** 若需要长期稳定 CLI，后续 change MUST 将可复用逻辑迁入包内 owner 并补 focused tests
