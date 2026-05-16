## ADDED Requirements

### Requirement: Snapshot workflow metadata
训练、验证和评估流程 MUST 在 snapshot next-frame baseline 的运行产物中记录该实验的无历史窗口语义。metadata MUST 足以让结果汇总工具区分 snapshot baseline 与历史窗口 baseline。

#### Scenario: 训练记录 snapshot metadata
- **WHEN** 用户训练 snapshot next-frame baseline
- **THEN** 运行 metadata MUST 记录 `variant: snapshot_next_frame`
- **AND** MUST 记录 `seq_len: 1` 和 `num_pred: 1`
- **AND** MUST 记录 `uses_history_window: false`
- **AND** MUST 记录 `uses_temporal_core: false`

#### Scenario: 评估报告记录 snapshot metadata
- **WHEN** 用户评估 snapshot next-frame baseline checkpoint
- **THEN** 评估报告 MUST 包含 checkpoint 或配置中的 snapshot metadata
- **AND** 报告 MUST 记录 enabled modalities、objective、scene、train/validation split CSV 和样本数
- **AND** 报告 MUST 标记 validation split 是 80/20 协议中的验证集合

### Requirement: Snapshot smoke workflow
项目 MUST 提供可通过统一训练入口运行的 snapshot smoke workflow。该 workflow MUST 使用 `conda run -n kd_mm_beam` 运行测试、训练或评估命令。

#### Scenario: 单模态 snapshot smoke test
- **WHEN** 开发者运行单模态 snapshot 配置的最小训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、validation 和 checkpoint 保存
- **AND** 日志中的模型配置 MUST 显示无 GRU representation core

#### Scenario: 多模态 snapshot smoke test
- **WHEN** 开发者运行五模态 snapshot fusion 配置的最小训练 smoke test
- **THEN** 训练流程 MUST 通过现有 fusion batch preparation 构造启用模态输入
- **AND** forward 输出 MUST 与 `num_pred=1` 的 labels 对齐
- **AND** 训练流程 MUST 不加载 teacher checkpoint

### Requirement: Snapshot 与历史窗口比较输出
实验工作流 MUST 允许用户在同一 Scenario 31 和同一 objective 下比较 snapshot baseline 与历史窗口 baseline。比较输出 MUST 明确展示实验变体和 split 协议，避免把不同时间上下文或不同窗口生成口径的结果混为同一条件。

#### Scenario: 记录 split 协议差异
- **WHEN** 用户对同一模态运行 snapshot baseline 和历史窗口 baseline
- **THEN** 两次运行的 metadata MUST 记录各自 train/validation CSV 路径和样本数
- **AND** 如果 CSV 路径或样本数不同，比较工具或文档 MUST 要求用户将其视为不同数据口径

#### Scenario: 结果表包含时间上下文
- **WHEN** 工具汇总 snapshot 与历史窗口结果
- **THEN** 表格或 JSON 输出 MUST 包含 `variant`、`seq_len`、`num_pred`、`uses_temporal_core` 和 `split_protocol`
- **AND** 模态强弱排序 MUST 能按这些字段分组计算
