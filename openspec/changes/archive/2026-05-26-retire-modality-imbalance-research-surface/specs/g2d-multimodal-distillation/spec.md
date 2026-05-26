## REMOVED Requirements

### Requirement: G2D result collection
**Reason**: `collect_multimodal_imbalance_results.py` 是多模态失衡研究汇总脚本；失衡研究线已退役。
**Migration**: 保留 G2D 训练期 diagnostics JSON。需要新的 G2D 结果汇总时，应在新的实验分析 capability 中定义输入和输出。

#### Scenario: G2D result collection 退役
- **WHEN** 用户运行 G2D 训练
- **THEN** 系统仍保存 G2D epoch diagnostics
- **AND** 系统不再要求提供 `tools/analysis/collect_multimodal_imbalance_results.py`
