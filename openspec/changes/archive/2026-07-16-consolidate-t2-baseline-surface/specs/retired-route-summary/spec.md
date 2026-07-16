## MODIFIED Requirements

### Requirement: 折叠退役路线不属于 current support surface
除 T2、S1、AMBER-Full、RMBP-MM 及其 MMW 运行闭包外，final C2、DeepSense6G 主线、CSI、physics-informed MMW、MMW GPS v2、Scene31-34 分析、C2/PCPG/BPRR/overnight launcher、旧 KD 和所有历史诊断/复现路线 MUST 只作为 retired 或 historical 说明出现。项目 MUST 不为这些路线提供 source module、current test、实体训练 YAML、registry entry、CLI、package facade、alias、migration guard 或 local runbook。

#### Scenario: 历史信息集中可查
- **WHEN** 维护者需要了解退役路线的用途
- **THEN** 当前历史说明 MUST 给出简短用途和 dated OpenSpec archive 或 git history 指针
- **AND** 系统 MUST 不要求旧模块、配置或命令仍存在

#### Scenario: T2 token 不会被误删
- **WHEN** cleanup 扫描 teacher、prototype、router、BPA、CMA 或 JEPA token
- **THEN** 它 MUST 保留 active T2 same-model consistency 与 ablation owner
- **AND** 仅删除不能追溯到 T2/baseline 的 legacy runtime 或 historical route
