## ADDED Requirements

### Requirement: 模态数据加载不依赖蒸馏配置
Dataset、batch preparation 和 label 对齐 MUST 由 experiment task、enabled modalities、prediction objective 和 supervised/adaptation workflow 决定。数据加载层 MUST 不读取 `distillation` 配置来决定 batch 字段或 label 语义。

#### Scenario: batch 构建忽略 distillation 字段
- **WHEN** 用户运行任一 supported supervised/adaptation 配置
- **THEN** batch preparation MUST 只根据 task 和 enabled modalities 构造输入
- **AND** 配置中若出现 `distillation` 字段 MUST 在配置解析阶段失败

