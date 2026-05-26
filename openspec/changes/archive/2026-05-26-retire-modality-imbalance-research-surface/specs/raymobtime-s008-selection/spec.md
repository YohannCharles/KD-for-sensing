## REMOVED Requirements

### Requirement: Raymobtime s008 模态失衡分析
**Reason**: Raymobtime s008 模态失衡分析入口和报告已退役；Raymobtime s008 本体仍保留为 current snapshot dataset/model/training/evaluation workflow。
**Migration**: 使用 Raymobtime s008 常规 metrics、`metrics.json` 和 `test_report.json` 进行结果查看；若未来需要新的 Raymobtime 分析，应提出独立 capability。

#### Scenario: Raymobtime s008 模态失衡分析退役
- **WHEN** 用户运行 Raymobtime s008 训练或评估
- **THEN** 系统不再要求输出单模态任务性能汇总、gate 均值、drop modality delta 或 LOS bucket 失衡表
