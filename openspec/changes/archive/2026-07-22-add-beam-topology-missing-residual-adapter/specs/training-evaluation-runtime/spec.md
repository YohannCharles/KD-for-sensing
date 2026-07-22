## ADDED Requirements

### Requirement: missing residual快速验证必须保持inner-only边界
训练与评估runtime MUST 允许从审计过的冻结clean cache训练轻量missing residual adapter，且 MUST 只使用inner-train拟合normalization、mean residual和loss calibration。Checkpoint MUST 只按最低有限inner-validation total adapter loss选择；本轮 MUST 标记single-seed、development和claim-ineligible。

#### Scenario: 完成missing residual矩阵
- **WHEN** 六组inner-only实验完成并生成comparison report
- **THEN** runtime MUST 停止且不得读取outer test或启动multi-seed
- **AND** 结果 MUST 不修改canonical recipe或正式claim

### Requirement: missing residual并行任务必须失败隔离
runtime MUST 在启动前记录GPU0-5状态，并 MUST 让六个任务独立记录PID、resolved config、日志与完成状态。单个任务失败 MUST 不终止其他任务，汇总 MUST 拒绝缺失结果。

#### Scenario: 某个adapter任务失败
- **WHEN** GPU0-5矩阵中一个worker返回非零状态
- **THEN** 其他worker MUST 继续直到各自结束
- **AND** launcher MUST 允许只重跑失败worker且不得静默跳过
