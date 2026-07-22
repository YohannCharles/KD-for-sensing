## ADDED Requirements

### Requirement: PGCD 仅作为 active T2 inner 开发任务
系统 MUST 允许 PGCD 在 active change 期间复用 MMW T2 四模态、BPA/topology、inner split 和训练 runtime，但 MUST 保持其为单 seed、development、claim-ineligible 的本地研究任务。PGCD MUST 不成为新的 public CLI、canonical 主方法或正式 evidence 输入。

#### Scenario: 干净 clone 加载 canonical recipe
- **WHEN** 用户加载 `configs/mmw/t2.yaml`、S1 或 baseline recipe
- **THEN** 系统 MUST 不实例化 PGCD 组件
- **AND** canonical config MUST 不依赖 PGCD outputs、checkpoint 或 generated config

#### Scenario: 运行 PGCD helper
- **WHEN** 用户显式 prepare 或 launch PGCD quick search
- **THEN** helper MUST 只生成 ignored local artifacts并追溯到本 active change
- **AND** 不得修改正式 claim 或 outer evidence
