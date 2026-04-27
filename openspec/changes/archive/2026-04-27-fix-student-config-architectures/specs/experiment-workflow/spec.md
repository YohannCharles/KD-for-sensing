## MODIFIED Requirements

### Requirement: 训练与评估行为等价
结构重构后，默认 image-only 和 image+radar 工作流 MUST 通过新脚本保持当前算法的核心行为语义，包括默认序列长度、预测步数、类别数、KD 模式、teacher 权重选择、student 架构选择、early stopping、gradient clipping、checkpoint 恢复和指标计算。

#### Scenario: 新配置默认参数
- **WHEN** 用户使用新脚本和默认配置启动训练或评估
- **THEN** 系统 MUST 使用从旧实现迁移而来的默认超参数语义，并保持相同的任务类型

#### Scenario: 默认 student 架构
- **WHEN** 用户使用默认 image-only 或 image+radar student 实验配置构建模型
- **THEN** 系统 MUST 为 image-only 工作流构建轻量 `image_student`
- **AND** 系统 MUST 为 image+radar 工作流构建轻量 `fusion_student`
- **AND** 默认 student 配置 MUST 与仓库提供的对应 `All_models/*Std*.pth` 权重结构兼容

#### Scenario: dry-run 训练
- **WHEN** 开发者使用 synthetic 或小比例数据运行一次短训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存的核心路径
