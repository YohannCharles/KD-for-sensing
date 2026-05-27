## 1. Execute Runner 与 CLI 语义

- [x] 1.1 新增窄的 LOSO execute runner 模块，接收现有 run plan、配置、输出目录和 overwrite/resume 选项，并返回结构化 execution result。
- [x] 1.2 修改 `kd-sensing-hist-beam-loso --execute`，使其在写出 `loso_plan.json` 后调用 execute runner；未传 `--execute` 时保持 plan-only 行为。
- [x] 1.3 调整 execute 返回 payload，确保 `execution.status` 只能是 `completed`、`failed` 或 `partial_failed`，不再在 execute 模式返回 `planned`。
- [x] 1.4 为每个 run/stage 写出 metadata，记录 fold、target scene、source scenes、variant、budget、seed、stage status、duration、artifact paths、checkpoint reuse 和失败原因。

## 2. Preflight 与 Quick Matrix

- [x] 2.1 将 quick validation 默认矩阵收敛为 target scene `34`、variants `v0_flat/v3_decoupled/v4_adapter/v5_adapter_proto/v6_full_finetune`、budgets `0/10`、seed `0`，DeepSense6G 默认 `data.dataset.portion=1.0`、`training.epochs=40`，并保留 CLI/config 缩小或扩展能力。
- [x] 2.2 实现 execute preflight，检查 source/target scene 数据根目录、必要 CSV、启用模态资源、输出目录可写性和矩阵配置合法性。
- [x] 2.3 让 preflight 失败在训练前停止执行，并输出包含缺失 scene、资源类型、路径和 run 组合的明确错误。
- [x] 2.4 保存 preflight metadata，记录已检查 scenes、CSV、启用模态、输出目录和实际 quick validation matrix。

## 3. Stage 编排与 Run 产物

- [x] 3.1 实现 source training stage wrapper，复用现有配置驱动 trainer 或训练 helper，产出标准 run directory、`metrics.json` 和 source checkpoint 路径。
- [x] 3.2 实现 source-only target_test evaluation stage，复用现有 evaluation/validator 与 HiST-Beam predictions writer，保存 `metrics.json` 和 target_test predictions。
- [x] 3.3 实现 `v4_adapter`、`v5_adapter_proto` 和 `v6_full_finetune` target adaptation stage，确保三者可复用同一 `v3_decoupled` source checkpoint 并记录 checkpoint 来源。
- [x] 3.4 实现 adapted target_test evaluation stage，保存 `metrics.json`、target_test predictions、adaptation checkpoint path 和 adaptation strategy metadata。
- [x] 3.5 在 adaptation metrics 或 metadata 中记录 trainable parameter count、total parameter count、trainable ratio、total adaptation time、可用时的 per-epoch time，以及 prototype coverage 或不可用原因。
- [x] 3.6 确保 predictions 至少包含 sample id、scene、true beam、predicted beam、top-k predictions、coarse true/pred、fine true/pred、variant metadata，并标记 split 为 `target_test`。

## 4. Summary 与快速验证结论

- [x] 4.1 实现 execute summary writer，输出 `loso_summary.json` 和 `loso_summary.csv`，汇总每个 run 的 stage status、metrics path、predictions path、checkpoint/prototype path、efficiency 指标和失败原因。
- [x] 4.2 实现 `quick_validation_conclusion.json` writer，比较 `v4_adapter`/`v5_adapter_proto` 相对 `v3_decoupled` source-only 的 Top-K、coarse/fine accuracy 和效率差异。
- [x] 4.3 在结论中比较 `v5_adapter_proto` 与 `v6_full_finetune` 的 accuracy、trainable ratio 和 adaptation time，并标明 adapter+prototype 是否优于 full fine-tuning。
- [x] 4.4 对缺失或失败 run 输出 `missing`/`inconclusive` 状态和原因，不得用 `0` 或默认数值伪造指标。

## 5. 测试与验证

- [x] 5.1 增加 `--execute` 单元测试，使用 fake stage executor 或最小 fixture 验证 execute 模式不再返回 `planned`，且 stage 顺序和 metadata 正确。
- [x] 5.2 增加无数据 preflight 测试，验证缺失数据根目录、CSV 或启用模态资源时在训练前明确失败。
- [x] 5.3 增加有最小测试数据的 smoke 测试，验证 execute 能跑出 `loso_summary.json`、`loso_summary.csv` 和 `quick_validation_conclusion.json`。
- [x] 5.4 更新或补充 CLI/config 测试，覆盖 quick validation 默认矩阵、target scene `34`、variants、budgets、seed 和用户缩小矩阵。
- [x] 5.5 运行 `openspec validate execute-hist-beam-quick-validation --strict` 并修复所有 OpenSpec 问题。
- [x] 5.6 运行 `openspec status --change execute-hist-beam-quick-validation`，确认 proposal、design、specs 和 tasks 均完成。
- [x] 5.7 运行 `conda run -n kd_mm_beam pytest tests/test_hist_beam_loso.py -q`，必要时补跑 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`。

## 6. Resource Probe 与中断可诊断性

- [x] 6.1 将 `quick_smoke` 调整为轻量资源探针，新增完整 `quick_validation` 配置保留 40 epoch、全量数据和最小方法验证矩阵。
- [x] 6.2 增加 CLI/config 的 `max_runs` 矩阵上限，并在 plan payload 中记录原始 planned run count 与实际 run count。
- [x] 6.3 在 execute runner 中实现 stage-start running metadata、执行进度事件和 `KeyboardInterrupt` partial summary，未启动 run 标记为 `missing`。
- [x] 6.4 为 source training/adaptation 写出 epoch 级 progress 文件，至少包含 epoch、总 epoch、耗时和可用 loss 统计。
- [x] 6.5 增加或更新测试，覆盖 resource smoke 配置、`max_runs`、stage-start metadata 和 interrupt partial summary。
- [x] 6.6 运行 `openspec validate execute-hist-beam-quick-validation --strict` 与相关 pytest。

## 7. Few-shot 标签解析与 adaptation 超参修复

- [x] 7.1 修复 few-shot sampler，使其优先使用 `future_beam_labelN`/`beam_labelN` 显式标签；当 CSV 字段是 beam-power 文件路径时，读取 power vector 并用 `argmax` 作为 beam label。
- [x] 7.2 确保 labeled sample manifest 记录真实 beam/coarse group 和 label 来源，失败时给出包含列名与路径的明确错误。
- [x] 7.3 为 `quick_validation` 配置显式设置 target adaptation epochs、entropy/prototype 权重，避免 0-label adaptation 静默不更新。
- [x] 7.4 增加测试覆盖整数标签、显式 label 列、power-vector 路径标签和 LOSO few-shot loader 的 budget>0 路径。
- [x] 7.5 运行 OpenSpec strict validate 与相关 pytest。
