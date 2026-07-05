## ADDED Requirements

### Requirement: Scripts lifecycle doctor
项目 MUST 提供 scripts lifecycle doctor，检查 tracked `scripts/` 和 `tools/analysis/` 文件是否具有明确 lifecycle 分类、owner、默认配置引用和输出边界。重复 package CLI 的 Python thin wrapper MUST 被标记为错误或高风险，除非 current spec 明确允许。

#### Scenario: 未分类脚本被发现
- **WHEN** tracked `scripts/` 下新增 Python 或 shell 文件，但 inventory、README/docs 或 OpenSpec 未登记其 lifecycle
- **THEN** scripts doctor MUST 报告未分类入口
- **AND** 报告 MUST 提示补充 lifecycle、输出边界和验证命令，或删除重复入口

#### Scenario: 脚本默认配置不存在
- **WHEN** local/manual runner 引用的默认 config path 不存在
- **THEN** scripts doctor MUST 报告失效引用
- **AND** 报告 MUST 不自动生成或恢复退役 config
