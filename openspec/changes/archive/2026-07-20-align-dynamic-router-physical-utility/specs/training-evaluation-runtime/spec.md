## ADDED Requirements

### Requirement: 物理效用筛选身份冻结
夜间动态 Router 决策对齐筛选 MUST 在启动前冻结 source checkpoint、source SHA、Joint panel SHA、loss source SHA、候选架构、决策目标、seed、batch、epoch 和 GPU 映射；已有 manifest 与请求不一致时 MUST fail closed。

#### Scenario: 八卡固定矩阵
- **WHEN** 启动完整 seed1 筛选
- **THEN** 系统在 GPU0--7 一卡一任务运行 `PATR/H2R × 四决策目标`，并为每项保存 resolved config、日志和状态

#### Scenario: 禁止身份漂移续跑
- **WHEN** 输出目录已有 manifest 且当前请求的任一冻结字段不同
- **THEN** launcher 拒绝复用该输出目录

### Requirement: Inner-only 晋级边界
决策对齐筛选结果 MUST 保持 inner-only 且 `claim_eligible=false`；只有使用冻结 Joint evaluator 通过既有材料性、置信区间与非劣 Gate 后才可规划 seed2--5。

#### Scenario: 夜间训练不自动形成正式 claim
- **WHEN** 八个 seed1 训练任务完成
- **THEN** 系统保留 checkpoint 和训练证据，但不得自动修改 canonical recipe、正式 claim 或 Gate
