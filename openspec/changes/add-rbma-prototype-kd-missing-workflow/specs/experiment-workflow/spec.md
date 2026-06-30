## ADDED Requirements

### Requirement: RBMA missing-modality ablation workflow
项目 MUST 提供配置驱动的 RBMA missing-modality ablation workflow。该 workflow MUST 通过当前训练/评估入口运行，不得新增根目录训练脚本、重复 trainer 或绕过配置解析 guard。

#### Scenario: 使用当前训练入口运行主配置
- **WHEN** 用户运行 RBMA prototype KD 主配置
- **THEN** 系统 MUST 通过 `conda run -n kd_mm_beam kd-sensing-train --config <config>` 或等价当前训练入口执行
- **AND** 系统 MUST 不要求用户运行根目录 `train.py`

#### Scenario: 配置覆盖不恢复 retired KD
- **WHEN** 用户通过 CLI 或 config override 尝试启用 `logits_kd`、`rkd`、`distillation.*` 或旧 teacher/student runtime
- **THEN** 配置加载 MUST 继续失败
- **AND** 错误信息 MUST 指向当前 U-MaskBeamJEPA full-to-partial stabilization 或当前 supervised/adaptation 入口

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
