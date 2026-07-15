## ADDED Requirements

### Requirement: On-disk script surface 必须完整分类
`scripts/` 下受控 `.py` 和 `.sh` 入口 MUST 被 lifecycle inventory 恰好覆盖一次，不论文件当前是否已被 Git 跟踪。每个 lifecycle 记录 MUST 包含 owner、保留原因、public/recommended relation、output boundary、focused validation 和 deletion condition。

#### Scenario: 未跟踪实验脚本未登记
- **WHEN** on-disk `scripts/` 出现未被任何 lifecycle 行匹配的 Python 或 shell 文件
- **THEN** architecture/compile guard MUST 失败并报告路径
- **AND** 文件 MUST 在删除或登记前不能通过 full verification

#### Scenario: Script 被多条规则匹配
- **WHEN** 某脚本同时匹配多个 lifecycle family 或精确行
- **THEN** guard MUST 失败并报告 duplicate classification

#### Scenario: 一次性 campaign 已完成
- **WHEN** local/manual script 的结论已进入 history/claim 且 deletion condition 成立
- **THEN** script MUST 从 current surface 删除或明确续期保留理由
- **AND** 系统 MUST NOT 为它新增 package wrapper
