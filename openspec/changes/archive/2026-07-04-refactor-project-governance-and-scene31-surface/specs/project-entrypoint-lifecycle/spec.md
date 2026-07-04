## ADDED Requirements

### Requirement: Scene31 local/manual 入口必须统一生命周期
Scene31 next-round、BC、beamsoft weak、funnel 和 magic overnight 的 runner、generator、summary 工具 MUST 被分类为 local/manual surface，并 MUST 不升级为 package CLI，除非新的 OpenSpec change 明确声明公开入口、输出边界和验证命令。

#### Scenario: runner 分类清晰
- **WHEN** 开发者检查 Scene31 shell 或 Python runner
- **THEN** 每个 runner MUST 在 inventory 或等价 lifecycle 文档中标注 owner、输入 manifest、默认输出 root、失败列表位置和删除/收敛条件

#### Scenario: package CLI 不被隐式新增
- **WHEN** Scene31 local/manual workflow 新增 runner 或 summary 能力
- **THEN** `pyproject.toml` MUST 不新增对应 `kd-sensing-*` console script
- **AND** README quickstart MUST 不把该 local/manual runner 写成长期 package 入口
