## ADDED Requirements

### Requirement: H2R 简化筛选身份不可变
训练 launcher MUST 在启动前冻结候选名、GPU、evidence profile、supervision、各 loss 权重、epoch、seed、batch、source checkpoint SHA、Joint panel SHA 和相关源码 SHA；已有 manifest 与请求不一致时 MUST fail closed。

#### Scenario: GPU0--7 并行启动
- **WHEN** 完整计划通过 preflight 且 GPU0--7 满足显存阈值
- **THEN** 系统一卡一任务并行启动八个 seed1 候选
- **AND** batch 固定为 64，候选输出互不覆盖

### Requirement: 固定 Joint 评估比较共同对照
每个完成候选 MUST 使用相同 81-condition Joint cache 比较 Uniform、train-fit static prior、frozen Current Router、Dynamic 和 Oracle，并记录 ADBA、normalized gain、relative delta、置信区间和受损块降权诊断。

#### Scenario: 训练后评估
- **WHEN** 八个训练任务完成
- **THEN** 系统按原 GPU 映射并行运行固定 Joint evaluation
- **AND** 任何缺失、失败或身份不匹配的候选不得进入完整结论
