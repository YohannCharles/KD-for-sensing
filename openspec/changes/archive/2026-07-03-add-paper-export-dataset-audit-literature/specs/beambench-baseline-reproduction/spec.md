## ADDED Requirements

### Requirement: Reproduction docs use current audit entrypoints
BeamBench/Arnold22 复现文档 MUST 使用当前存在的 package/module/script 入口描述数据审计、mock smoke、官方 blocked 审计和本地 substitute。文档 MUST 不把不存在的旧脚本作为当前推荐命令。

#### Scenario: README_REPRODUCE 不推荐缺失脚本
- **WHEN** `README_REPRODUCE.md` 给出数据检查、mock smoke 或官方评估计划命令
- **THEN** 命令 MUST 指向当前存在的 package CLI、`python -m kd_sensing...` 模块或保留脚本
- **AND** 如果保留历史旧脚本名，MUST 明确标记为 historical/unavailable

#### Scenario: audit 输出 official blocked reason
- **WHEN** BeamBench official data、weights、source/config 或 environment 不完整
- **THEN** 复现报告 MUST 使用 dataset/reproducibility audit 输出 blocked reason
- **AND** 报告 MUST 不填入伪造 official reproduction 数值

### Requirement: Paper export consumes BeamBench claim status
BeamBench/Arnold22 本地 substitute、official blocked、upper-bound 和 historical ablation MUST 能被 paper export 按状态过滤。

#### Scenario: official blocked 不进入结果主表
- **WHEN** paper export 读取 BeamBench blocked official claim
- **THEN** blocked row MUST 不进入 main results table
- **AND** 它 MAY 进入 reproducibility appendix，并保留 blocked reason

#### Scenario: local substitute 保留 caveat
- **WHEN** paper export 读取 Arnold22 local substitute row
- **THEN** 输出表格 MUST 保留 local substitute status、target source、selection split、metric profile 和 official caveat
