## ADDED Requirements

### Requirement: Scene31 next-round local follow-up workflow
项目 MUST 将 Scene31 next-round follow-up 作为 local/manual experiment workflow 处理。该 workflow MUST 复用现有 `kd-sensing-train`、missing-pattern fresh eval 和本地输出边界，不得改变已有 Scene31 es20 night-grid 配置或 baseline 行为。

#### Scenario: next-round fresh eval 查找配置
- **WHEN** fresh eval 需要评估 next-round manifest 中的 run
- **THEN** 配置查找 MUST 支持 `configs/scene31/next_round/<run>.yaml`
- **AND** 仍 MUST 继续支持已有 `configs/scene31/night_grid/<run>.yaml` 与 `configs/scene31/<run>.yaml`

#### Scenario: local/manual 输出边界
- **WHEN** 用户运行 Scene31 next-round launcher 或汇总脚本
- **THEN** 训练、评估和汇总产物 MUST 写入 ignored 的 `outputs/` 或 `logs/` 下
- **AND** 系统 MUST 不提交 checkpoint、日志、fresh eval CSV 或训练输出
