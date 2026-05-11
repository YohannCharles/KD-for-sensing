## 1. Phase 1.5 清单与入口

- [x] 1.1 新增 Phase 1.5 默认运行清单，声明现有 `scene32_marf/conditional_utility` 输入、`best_top1/best/last` checkpoint roles、5 个 fixed-subset baseline、3 个 seed、输出目录和判定阈值。
- [x] 1.2 新增 Phase 1.5 CLI 或 analysis script，能够读取运行清单并创建 `outputs/scene32/phase_1_5_utility_validation/` 目录。
- [x] 1.3 在输出 metadata 中保存解析后的清单、输入文件路径、checkpoint role、seed、subset、run name 和缺失产物状态。

## 2. Bootstrap 统计

- [x] 2.1 实现读取 `subset_predictions` 与 `conditional_utility_per_sample_delta` 的 paired delta helper，覆盖 weak-plus 和 `all - strong_only` comparison。
- [x] 2.2 实现 cluster bootstrap 95% CI，优先使用 `seq_id`，缺失时 fallback 到 `sample_id` 或 `dataset_index` 并记录 fallback metadata。
- [x] 2.3 输出 `conditional_utility_bootstrap_ci.csv`，包含 comparison、weak modality、metric、horizon、mean delta、CI、bootstrap 次数、cluster 数和 cluster key。
- [x] 2.4 将关键 CI 写入 Phase 1.5 summary，并确保 tiny positive gain 在 CI 跨 0 时被标记为不显著。

## 3. Checkpoint Matrix 复核

- [x] 3.1 实现 checkpoint role 解析，支持 `best_top1.pth`、`best.pth`、`last.pth`，并兼容未来显式 `best_dba.pth`。
- [x] 3.2 对缺失的 checkpoint audit 产物生成可执行命令，命令必须使用 `conda run -n kd_mm_beam python tools/analysis/run_conditional_utility_audit.py ...`。
- [x] 3.3 汇总每个 checkpoint role 的 subset metrics、marginal utility、oracle gain、teacher complementarity 和 diagnosis。
- [x] 3.4 输出 checkpoint comparison 表，并标记 weak utility 结论是否跨 checkpoint 一致。

## 4. Dedicated Fixed-Subset Baseline

- [x] 4.1 定义 5 个 dedicated subset：`gps+mmwave`、`gps+mmwave+image`、`gps+mmwave+radar`、`gps+mmwave+lidar`、`all`，并保持中心模态顺序。
- [x] 4.2 生成 5 subsets x 3 seeds 的训练命令，命令必须使用 `conda run -n kd_mm_beam`，并覆盖 seed/run_name 以避免输出互相覆盖。
- [x] 4.3 汇总 baseline metrics，输出每个 subset 的 Top1、Top3、DBA、loss 的 `mean ± std`，同时包含 `t+1/t+2/t+3/avg`。
- [x] 4.4 将 dedicated `gps+mmwave` 固定为主基线，禁止用 MARF masking `strong_only` 替代 final baseline conclusion。

## 5. 决策报告与 Diagnosis

- [x] 5.1 更新 conditional utility diagnosis，使 global useful 至少满足配置的最小效果量阈值和 bootstrap CI 下界大于 0。
- [x] 5.2 更新 conditional useful 判定，使 bucket/horizon 证据记录样本数、均值、CI、阈值和触发来源。
- [x] 5.3 生成 Phase 1.5 总报告，汇总 bootstrap、checkpoint matrix、baseline matrix、bucket highlights、teacher complementarity、oracle gain 和路线建议。
- [x] 5.4 报告中明确区分最终证据与探索性证据：baseline 或 CI 不完整时只能输出 pending，不得给最终路线结论。

## 6. 验证

- [x] 6.1 新增 bootstrap helper 单元测试，覆盖 paired delta、CI 形状、cluster fallback 和 all-modal comparison。
- [x] 6.2 新增 Phase 1.5 manifest/report 测试，覆盖缺失产物 pending、metadata 记录和 dedicated strong-only 主基线选择。
- [x] 6.3 新增 diagnosis 阈值测试，确认 tiny positive delta 或 CI 跨 0 不会被标记为 globally useful。
- [x] 6.4 运行定向测试：`conda run -n kd_mm_beam pytest -q tests/test_phase_1_5_utility_validation.py tests/test_conditional_utility_metrics.py`。
