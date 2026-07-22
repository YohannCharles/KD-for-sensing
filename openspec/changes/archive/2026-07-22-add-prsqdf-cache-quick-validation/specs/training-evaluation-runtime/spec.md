## ADDED Requirements

### Requirement: 冻结 C0 的 cached-quality 训练必须保持 validation 与证据边界
runtime MUST 允许 PR-SQDF从审计过的本地 block cache训练独立 quality head，且 MUST 不运行或更新 semantic backbone。checkpoint MUST 只由 inner validation fused beam loss选择；evaluation、weather和corruption结果不得用于重调lambda、模型选择或风险归一化。

#### Scenario: 选择 PR-SQDF checkpoint
- **WHEN** Q1--Q5完成 cache epoch
- **THEN** best checkpoint MUST 对应最低有限 inner validation fused beam loss
- **AND** checkpoint provenance MUST 记录 C0 SHA、cache manifest SHA、train normalization SHA、seed和claim-ineligible状态

### Requirement: PR-SQDF 六卡任务必须独立记录状态
runtime MUST 将 Q0--Q5分别映射到 GPU0--5，保存 PID、resolved config、日志、checkpoint/metrics和失败原因；单任务失败 MUST 不终止其他任务，且 runtime MUST 不发送信号给无关 GPU进程。

#### Scenario: 一个 quality方向失败
- **WHEN** 某个任务以非零退出码结束
- **THEN** 其状态 MUST 标记 failed并记录退出码
- **AND** 其余任务 MUST 继续等待完成
- **AND** 修复后只能显式重跑失败任务
