## MODIFIED Requirements

### Requirement: public 与本地入口保持最小闭包

系统 MUST 只注册 train、evaluate、preprocess 三个 console script。仓库内脚本 MUST 仅保留 MMW all-weather、BPA/CMA、必要 summary 和 compile verification；CMSBL MUST 复用 train/evaluate 入口，不增加 thin wrapper 或本地 runner。

#### Scenario: 枚举入口

- **WHEN** 架构测试扫描 CLI 与 `scripts/*.py`
- **THEN** public console scripts MUST 精确等于三个核心入口
- **AND** 每个 retained script MUST 能追溯到 current T2/baseline 或验证 owner
