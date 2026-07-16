# retired-route-summary Specification

## Purpose

规定退役路线的唯一历史记录方式，确保历史用途可由集中说明和 archive 追溯，但不保留运行代码、配置、命令、alias 或迁移层。

## Requirements

### Requirement: 折叠退役路线不属于 current support surface

除 T2、S1、AMBER-Full、RMBP-MM 及其 MMW 运行闭包，以及 DeepSense6G Scene31–34 的受限 T2 四模态数据路径和其传递运行依赖外，final C2、CSI、physics-informed MMW、MMW GPS v2、历史 DeepSense6G CSI/mmWave/input-beam/soft-label/cache/场景别名分支、Scene31–34 历史分析、C2/PCPG/BPRR/overnight launcher、旧 KD 和所有历史诊断/复现路线 MUST 只作为 retired 或 historical 说明出现。项目 MUST 不为这些路线提供 source module、current test、实体训练 YAML、registry entry、CLI、package facade、alias、migration guard 或 local runbook。

#### Scenario: 历史信息集中可查

- **WHEN** 维护者需要了解某条已删除路线
- **THEN** 当前历史说明 MUST 给出简短用途和 dated OpenSpec archive 或 git history 指针
- **AND** 系统 MUST 不要求旧模块、配置或命令仍存在

#### Scenario: T2 token 不会被误删

- **WHEN** cleanup 扫描 teacher、prototype、router、BPA、CMA 或 JEPA token
- **THEN** 它 MUST 保留 active T2 same-model consistency 与 ablation owner
- **AND** 仅删除不能追溯到 current T2/baseline 或 DeepSense6G T2 闭包的 legacy runtime 或 historical route

### Requirement: 历史说明不构成兼容承诺

历史说明 MUST 不提供可运行命令、YAML 映射或 compatibility stub。

#### Scenario: 请求退役路径

- **WHEN** 用户加载旧配置或导入旧模块
- **THEN** 普通缺失文件或 unknown-name 错误即可
- **AND** 系统 MUST 不自动迁移或构建替代路径
