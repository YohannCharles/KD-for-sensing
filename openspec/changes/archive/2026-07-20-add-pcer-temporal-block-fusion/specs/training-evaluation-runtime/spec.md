## ADDED Requirements

### Requirement: PCER quick-validation 使用验证集最佳 checkpoint
训练 runtime MUST 以 opt-in 配置逐 epoch运行 validation，并在 validation loss 改善时发布独立 `best.pth`。默认 fixed-epoch `last.pth` 行为 MUST 不变，测试集 MUST 不参与 checkpoint 选择。

#### Scenario: 选择 quick-validation checkpoint
- **WHEN** PCER 训练在多个 epoch 产生 validation loss
- **THEN** `best.pth` MUST 对应最低有限 validation loss并记录 epoch
- **AND** 最终固定 mask 评测 MUST 加载该 checkpoint

### Requirement: PCER 固定评测按样本身份生成 mask
evaluation runtime MUST 以 global eval seed、stable sample identity、mask type 和 variant 生成逐样本固定 mask，并在 A0--A3 间校验 identity 一致。S1 MUST 使用三个 variant；S3 MUST 分别覆盖每个模态；S5 MUST 在测试集上均衡合法模态对和 recent burst template。

#### Scenario: 固定六场景评测
- **WHEN** evaluator 运行 S0--S5
- **THEN** MUST 输出 Top-1、Top-3、Top-5、Within-3、beam-index MAE 和已有通信指标
- **AND** S3 MUST 输出每模态、macro 和 worst，S5 MUST 输出 macro 和 worst pair

### Requirement: quick-validation 证据保持开发边界
四组 PCER quick-validation MUST 标记为单 seed、inner/development、claim-ineligible。系统 MUST 不把本轮结果写入正式 claim，也 MUST 不自动运行多 seed 或剩余实验矩阵。

#### Scenario: 汇总完成
- **WHEN** comparison report 已生成
- **THEN** runtime MUST 停止在四组结果和预注册判断
- **AND** 下一批实验只能作为建议，不得自动启动
