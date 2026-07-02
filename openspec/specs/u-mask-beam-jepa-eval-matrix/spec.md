# u-mask-beam-jepa-eval-matrix Specification

## Purpose
定义 U-MaskBeamJEPA 在固定和随机缺失模态条件下的评估矩阵、reliability diagnostics、结果导出和非侵入集成边界。

## Requirements
### Requirement: Fixed Missing Pattern Evaluation
系统 SHALL support evaluation under named fixed missing patterns. 固定 missing mask MUST use `1 = modality available` and `0 = modality missing`, and MUST NOT generate all-missing masks unless explicitly allowed by a future opt-in configuration.

#### Scenario: 默认固定缺失矩阵
- **WHEN** 开发者请求默认 fixed patterns 并提供模态列表
- **THEN** 系统 MUST 返回稳定、可读的 pattern 名称，至少覆盖 full、single missing、only one available 和 pair missing

#### Scenario: 固定 pattern 评估
- **WHEN** U-MaskBeamJEPA evaluation matrix 执行一个 named fixed pattern
- **THEN** 系统 MUST 显式构造该 pattern 的 `missing_mask` 并传入模型 forward

### Requirement: Random Missing Evaluation
系统 SHALL support random missing evaluation with configurable `p_missing`.

#### Scenario: 随机缺失率评估
- **WHEN** evaluation matrix 配置 `random_missing` 缺失率列表
- **THEN** 系统 MUST 对每个 batch 重新采样 mask，并以 `random_p<rate>` 形式记录 pattern 名称

#### Scenario: 至少一个模态可用
- **WHEN** random missing evaluation 启用 `ensure_at_least_one`
- **THEN** 系统 MUST 保证每个样本至少一个模态可用

### Requirement: Reliability Diagnostics
系统 SHALL report global reliability and modality reliability statistics per pattern.

#### Scenario: reliability 与错误统计
- **WHEN** evaluation matrix 完成一个 pattern
- **THEN** 系统 MUST 输出 mean confidence、mean global reliability、correct/wrong global reliability、mean modality reliability 和 available-modality reliability 字段，缺失字段 MUST 使用稳定 schema 表示为 NaN 或等价空值

### Requirement: Exportable Results
系统 SHALL export results to CSV and JSON, and MAY export Markdown.

#### Scenario: 结果导出
- **WHEN** evaluation matrix 返回 pattern-level results
- **THEN** 系统 MUST 能保存 `eval_matrix.csv` 和 `eval_matrix.json`，并 MAY 保存 Markdown summary table

### Requirement: Non-invasive Integration
系统 SHALL NOT change training-time behavior or model architecture.

#### Scenario: 评估入口不影响训练
- **WHEN** 新增 evaluation matrix 工具、CLI、配置和测试
- **THEN** 系统 MUST NOT 修改 U-MaskBeamJEPA 模型主干、训练 loss、encoder、训练 extension 随机 mask 行为或既有训练入口

### Requirement: Missing pattern evaluation workflow
项目 MUST 支持按 missing pattern 运行 evaluation，并将 pattern 名称、mask、样本数和指标写入报告。该 workflow MUST 复用当前 eval matrix 或包内 CLI 边界。

#### Scenario: 指定 eval patterns
- **WHEN** 用户指定 `full missing_image missing_radar missing_lidar missing_gps non_gps_only only_gps random_0.25 random_0.5 random_0.75`
- **THEN** evaluation MUST 为每个 pattern 构造确定性或配置声明的 mask
- **AND** report MUST 按 pattern 输出 top1、top5、loss 和样本数

#### Scenario: pattern eval 不修改原 batch
- **WHEN** evaluation 构造某个 missing pattern
- **THEN** 系统 MUST 只把 missing mask 传给模型或 runtime
- **AND** 原 batch 中的模态 tensor MUST 不被原地修改
