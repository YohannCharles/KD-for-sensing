## ADDED Requirements

### Requirement: runtime 复用 training extension 保存 CMSBL 状态

runtime MUST 在 epoch 结束、checkpoint 写出前聚合 train-only capacity/mask 状态并写出单一 JSON。resume MUST 恢复相同 reference identity、EMA、count 和 initialized 状态；validation/test MUST 只读模型与状态。

#### Scenario: epoch 结束保存 last checkpoint

- **WHEN** CMSBL 训练 epoch 完成
- **THEN** runtime MUST 先更新长期状态和 JSON，再保存 `last.pth`
- **AND** 恢复后 epoch accumulator MUST 为空

### Requirement: CMSBL 继续使用现有训练与评估入口

系统 MUST 通过 canonical T2 配置、strict overrides、现有 train CLI 和 fixed-mask evaluator运行 CMSBL。系统 MUST 不增加第二 trainer、CMSBL console script 或自动 GPU runner。

#### Scenario: 评估 CMSBL checkpoint

- **WHEN** 现有 evaluator 读取一个 CMSBL checkpoint
- **THEN** 输出 MUST 保留 15-pattern、Full/Single/Double/Triple/All-14 macro/worst、Top-1/3/5、Within-3、MAE 和 ADBA
- **AND** 不得更新 capacity/mask 训练状态
