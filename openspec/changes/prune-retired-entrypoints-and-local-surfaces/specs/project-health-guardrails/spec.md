## ADDED Requirements

### Requirement: Guardrails validate pruned surface from source of truth
架构边界测试 MUST 从 `pyproject.toml`、真实 tracked paths、README/current docs、inventory lifecycle 和集中 retired-route guard 推导检查，不得维护完整脚本 allowlist、完整 tombstone 文件清单或完整 config 数据库镜像。

#### Scenario: Scripts checked from inventory or manifest
- **WHEN** tracked scripts 存在
- **THEN** 架构边界测试 MUST 验证每个脚本有 lifecycle 记录或由 retained generator/manifest 推导
- **AND** 测试 MUST 不复制完整脚本文案说明

#### Scenario: Retired route checked centrally
- **WHEN** retired route token 出现在 docs/specs/source/tests 中
- **THEN** 测试 MUST 验证其语境是 retired、historical、guard 或 migration
- **AND** 测试 MUST 不要求每条 retired route 拥有独立 tombstone spec

### Requirement: Validation covers deletion batches
每个删除批次 MUST 至少运行对应 focused validation：入口删除运行 CLI/architecture checks，config 删除运行 config characterization/generator checks，tombstone 折叠运行 OpenSpec strict 和 retired-route guard tests。

#### Scenario: Deletion batch validation
- **WHEN** 本 change 删除 CLI、script、config、spec 或 test 文件
- **THEN** 最终说明 MUST 记录已运行的 focused validation
- **AND** 未运行的验证 MUST 说明原因和剩余风险
