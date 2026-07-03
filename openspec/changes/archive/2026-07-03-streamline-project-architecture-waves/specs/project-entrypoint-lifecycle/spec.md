## ADDED Requirements

### Requirement: Scripts are classified before retention
`scripts/` 和 `tools/analysis/` 中保留或新增的入口 MUST 明确分类为数据准备、研究诊断、shell orchestration、local/manual experiment helper 或 package CLI 缺口补充。重复 package CLI 的 Python thin alias MUST 删除；local/manual helper MUST 不被 README、AGENTS、docs 或 OpenSpec 写成长期推荐入口。

#### Scenario: 新脚本有 lifecycle
- **WHEN** 本 change 保留、新增或修改 `scripts/*.py`、`scripts/**/*.py` 或 shell orchestration
- **THEN** inventory 或 tasks MUST 记录该脚本 owner、lifecycle、是否 local/manual、输出边界和替代 package CLI
- **AND** 架构边界测试 MUST 拒绝未分类长期脚本入口

#### Scenario: Thin alias 被删除
- **WHEN** 脚本只解析参数后调用已有 package CLI 或包内 CLI 的同名 main
- **THEN** pyproject console script 或包内 CLI MUST 成为推荐入口
- **AND** 该 thin alias MUST 删除或被明确标注为短期 local/manual helper 并登记删除条件

### Requirement: Local experiment orchestration cannot become hidden public API
本地批量实验、night-grid、next-round、seed sweep、fresh eval 汇总或类似脚本 MAY 保留为 local/manual workflow，但必须声明不作为稳定 public API。它们 MUST 写入 ignored outputs/logs，且不得提交 checkpoint、metrics、fresh eval 结果或真实运行产物。

#### Scenario: Local manual runner
- **WHEN** local/manual runner 生成或消费 Scene31、RBMA、BTAPA、night-grid 或 next-round 配置
- **THEN** 脚本 MUST 支持 dry-run 或无副作用 sanity path
- **AND** 文档 MUST 指向输出边界并说明真实训练产物不提交

### Requirement: CLI glue stays thin
Package CLI 文件 MUST 只承担参数解析、配置覆盖、轻量 IO、调用 owner module 和 user-facing exit code。真实 workflow、training loop、evaluation loop、dataset preparation、benchmark suite 或 report builder MUST 位于 owner module。

#### Scenario: 修改 package CLI
- **WHEN** 本 change 修改 `src/kd_sensing/cli/` 下入口
- **THEN** CLI 文件 MUST 不复制训练、评估、dataset parsing 或 benchmark aggregation 主逻辑
- **AND** 对应 `kd-sensing-* --help` 或包内 CLI smoke MUST 继续可运行

