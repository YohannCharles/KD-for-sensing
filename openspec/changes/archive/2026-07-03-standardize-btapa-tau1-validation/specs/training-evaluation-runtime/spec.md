## ADDED Requirements

### Requirement: apples-to-apples checkpoint 复评
系统 MUST 提供只读 apples-to-apples 复评入口，用同一个 checkpoint 加载、missing pattern 构造、evaluation pass 和指标计算逻辑复评指定 run。复评入口 MUST 支持 `best_val_top1`、`latest`、`best_avg_missing_top1` 和 `manual_path` checkpoint 选择策略，并 MUST 输出 metrics CSV、Markdown、delta CSV 和 checkpoint manifest。

#### Scenario: 复评指定 runs
- **WHEN** 用户运行 `scripts/reevaluate_apples_to_apples.py` 并传入 root、runs、eval patterns、checkpoint policy 和 out_dir
- **THEN** 系统 MUST 为每个找到 checkpoint 的 run 重新 load checkpoint 并执行评估
- **AND** 输出 metrics 行 MUST 至少包含 run name、checkpoint path、checkpoint epoch、pattern、Top-1/Top-3/Top-5、ADBA、MAE、loss 和 count

#### Scenario: checkpoint 缺失
- **WHEN** 指定 run 找不到符合策略的 checkpoint
- **THEN** 系统 MUST 打印 warning
- **AND** checkpoint manifest 与 metrics 表 MUST 标记 `missing_checkpoint`

#### Scenario: radar_only 口径差异告警
- **WHEN** old V3 与 current proto baseline 的 `radar_only` Top-1 差异超过 5 个百分点
- **THEN** 系统 MUST 打印 `[WARN] radar_only metric mismatch is large; check pattern construction or checkpoint selection`

### Requirement: early-stopped run 状态识别
summary 和训练产物 MUST 能区分跑满完成、early stopping 正常退出、有 checkpoint 但未完成、以及失败或被 kill 的 run。early stopping 判据 MUST 支持 metrics、训练日志或 run metadata 中的显式标记。

#### Scenario: early stopped 不标为失败
- **WHEN** run 未达到 expected epochs 但存在 `early_stopped=true`、`Early stopping triggered` 或等价 metadata
- **THEN** summary MUST 将状态标记为 `completed_early_stopped`
- **AND** 输出 MUST 包含 best epoch、final epoch、early stop epoch、early stopped 和 expected epochs 字段

#### Scenario: checkpoint 存在但无 early stop 标记
- **WHEN** run 存在 checkpoint 但 final epoch 小于 expected epochs 且没有 early stop 标记
- **THEN** summary MUST 标记为 `incomplete_has_checkpoint`
- **AND** 不得把该状态与 `killed_or_failed` 混淆

### Requirement: 统一 checkpoint resolver
系统 MUST 提供统一 checkpoint resolver，供复评、分析和 summary 脚本选择 checkpoint。resolver MUST 支持 `manual_path`、`best_val_top1`、`latest` 和 `best_epoch_from_metrics` 策略，manual path 优先级最高；无法解析时 MUST 返回清晰 warning，不得静默选择其它 run 的 checkpoint。

#### Scenario: seed run 不被前缀误导
- **WHEN** 同时存在 `main_v3_strong_reliability_btapa_tau1` 与 `main_v3_strong_reliability_btapa_tau1_seed2` checkpoint
- **THEN** resolver MUST 只选择 sidecar、run_dir 或 config_slug 属于目标 run 的 checkpoint
- **AND** `best_val_top1` MAY 从 metrics、sidecar metric 或文件名 `_primary_acc_*.pth` 解析数值，但不得只靠 glob 顺序

#### Scenario: manual path 优先
- **WHEN** 调用方传入 `manual_path`
- **THEN** resolver MUST 返回该路径和可解析 epoch
- **AND** 若路径不存在 MUST 返回 warning 并保持 path 为空

### Requirement: eval consistency debug 报告
系统 MUST 提供 `scripts/debug_eval_consistency.py`，按 root、run、checkpoint、patterns 和 out_dir 执行 fresh eval，并输出 JSON/Markdown 报告。报告 MUST 包含 checkpoint path/epoch、config path、eval dataset path/split、metrics.csv val_acc、full/avg_missing fresh top1、标准 pattern mask、每个 pattern 样本数，以及 `abs(val_acc - full_top1) > 0.03` 的 warning。

#### Scenario: full top1 与 val_acc 不一致
- **WHEN** fresh full-pattern top1 与 metrics.csv 选中 checkpoint epoch 的 val_acc 差值超过 0.03
- **THEN** 报告 MUST 写入 warning
- **AND** 报告 MUST 同时记录 evaluation split 与训练 validation split 线索，帮助判断是否 split 不一致
