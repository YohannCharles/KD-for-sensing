## 1. 基线盘点

- [x] 1.1 汇总现有 s008 seed42 单任务和多任务 run，生成 run inventory，记录 run_dir、config、resolved_config、objective、modalities、seed、split metadata 和核心 metrics。
- [x] 1.2 检查现有 `outputs/raymobtime_s008/analysis/s008_single_task_sensing_only_seed42` 中 gate、drop、gradient 和 LOS bucket 表是否为空，并记录缺失原因。
- [x] 1.3 用 `conda run -n kd_mm_beam python scripts/train.py --help` 和现有 Raymobtime 配置确认训练入口、配置覆盖方式和输出目录仍可用。

## 2. s008 实验矩阵配置

- [x] 2.1 为 sensing-only 单任务矩阵整理配置或配置覆盖，覆盖 `coord`、`image`、`lidar`、`coord+image+lidar` × beam/LOS/link。
- [x] 2.2 为 selection multitask 任务组合消融整理配置或配置覆盖，覆盖 `beam_only_multitask_model`、`beam+los`、`beam+link`、`beam+los+link`。
- [x] 2.3 为 loss 权重消融整理配置或配置覆盖，覆盖 original、equal 和 beam-heavy 三类权重。
- [x] 2.4 确保所有矩阵配置固定 s008 cache、split、portion、输入尺寸和 sensing-only modality 口径，并将 run 输出写入 ignored 目录。

## 3. s008 运行与复评估

- [x] 3.1 使用 `conda run -n kd_mm_beam ...` 先运行 seed42 的任务组合和 loss 权重消融，并保存 resolved_config、metrics、train_log 和 checkpoint registry。
- [x] 3.2 将关键对照扩展到 seed `7` 和 `123`，保持数据 split 不变，只改变训练随机 seed。
- [x] 3.3 对每个 multitask run 汇总 best `val_selection_multitask_loss`、best `val_beam_top1` 和 best `val_link_mae` 对应 epoch。
- [x] 3.4 若 checkpoint 可用，使用 `conda run -n kd_mm_beam ...` 对 best-by-metric checkpoint 在同一 validation split 上复评估；若不可用，在报告中标记只能使用 train_log 视角。

## 4. 内部诊断

- [x] 4.1 对 task-aware multitask run 生成按 task/modality 的 gate 均值与标准差表。
- [x] 4.2 对多模态 run 生成 drop `coord`、drop `image`、drop `lidar` 的 test-time modality drop delta 表。
- [x] 4.3 生成按 task/modality 的 gradient norm 或 contribution 表；若当前代码无法生成，记录阻塞点和所需实现。
- [x] 4.4 生成按 LOS bucket 分组的 beam Top-K 和 `beam_dba_current` 对比表。

## 5. 判定报告

- [x] 5.1 生成 s008 run matrix CSV、metric comparison CSV、diagnostic tables、summary JSON 和 markdown 判定报告。
- [x] 5.2 在报告中按 `confirmed_imbalance`、`likely_parameter_issue`、`inconclusive` 或 `diagnostics_blocked` 给出单一结论。
- [x] 5.3 在报告中明确反证检查：early stopping 是否解释 beam 退化、loss 权重是否能恢复 beam、task-combo 是否定位具体冲突任务、诊断证据是否支持模态/任务支配。
- [x] 5.4 确认报告不引用本地 checkpoint、cache 或 TensorBoard 文件作为源码变更内容。

## 6. s009 外部验证门槛

- [x] 6.1 仅当 s008 报告达到 `confirmed_imbalance` 或高置信 `inconclusive` 时，检查本地 s009 数据契约、cache、label、split 和 modality 是否具备最小复刻条件。
- [x] 6.2 若 s009 需要新增 dataset 或 preprocess 能力，停止本 change 的 s009 执行并创建独立 OpenSpec change。（本轮 s008 结论为 `likely_parameter_issue`，未进入 s009，故无需新增独立 change。）
- [x] 6.3 若 s009 已具备条件，复刻最小矩阵：`lidar` 与 `coord+image+lidar` 的 beam/LOS/link 单任务、original multitask、beam-heavy multitask 和最可疑 task-combo。（未触发：s008 未达到进入 s009 的门槛。）
- [x] 6.4 生成 s009 外部验证附录，明确它只用于跨场景验证，不替代 s008 的失衡确认。

## 7. 校验

- [x] 7.1 运行 `openspec validate diagnose-raymobtime-s008-modality-imbalance --strict`。
- [x] 7.2 若新增或修改实验脚本/分析脚本，运行相关单元测试，并使用 `conda run -n kd_mm_beam pytest <相关测试> -q`。
- [x] 7.3 若涉及训练入口、评估入口或 Raymobtime 分析入口，运行对应 `--help` 命令并确认命令正常退出。
