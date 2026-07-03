## 1. Missing Pattern 与统一复评

- [x] 1.1 修正并扩展统一 missing pattern helper，提供 canonical 顺序、alias、mask/name/list API，并迁移相关脚本复用。
- [x] 1.2 新增 `scripts/reevaluate_apples_to_apples.py`，支持 checkpoint policy、统一 force-mask evaluation、CSV/Markdown/delta/manifest 输出和结论告警。
- [x] 1.3 用轻量测试或 dry-run 验证 missing pattern helper 与 apples-to-apples 参数解析；项目 Python 命令使用 `conda run -n kd_mm_beam`。

## 2. BTAPA tau1 seed 与分析

- [x] 2.1 新增 BTAPA tau1 seed2/seed3 配置，除 seed 和输出路径外保持原 tau1 配置一致。
- [x] 2.2 新增 `scripts/analyze_btapa_tau1_seeds.py`，读取已有 run 指标并输出 seed metrics、mean±std、Markdown 和 delta-vs-proto mean。
- [x] 2.3 增强 `scripts/analyze_btapa_runs.py`，支持 `--candidate` 和保守 paper-ready observation。

## 3. es20、summary 与 launcher

- [x] 3.1 新增 BTAPA tau1 es20、es20 seed2、es20 seed3 配置，启用 20 epoch 与 early stopping 字段。
- [x] 3.2 必要时补训练 runtime 的轻量 early stopping metadata 写出，确保 metrics/train log 能记录 early-stopped 状态。
- [x] 3.3 修改 `scripts/summarize_missing_runs.py`，区分 completed、completed_early_stopped、incomplete_has_checkpoint 和 killed_or_failed。
- [x] 3.4 新增 `scripts/run_btapa_tau1_validation.sh`，默认串行、支持 dry-run/skip 参数和独立训练日志。

## 4. 验证

- [x] 4.1 运行 `openspec validate standardize-btapa-tau1-validation --strict`。
- [x] 4.2 运行脚本 help/dry-run/config smoke 等可行验证；项目 Python 命令使用 `conda run -n kd_mm_beam`。
- [x] 4.3 汇总未运行的长训练或依赖真实 checkpoint/dataset 的验证项及原因。

## 5. proto vs BTAPA tau1 复核补强

- [x] 5.1 新增统一 checkpoint resolver，并迁移 apples-to-apples、BTAPA seed 分析和 summary 的 best checkpoint 选择。
- [x] 5.2 补齐 missing pattern 标准 API 与模型实际顺序 mask 映射，确保 eval/analysis 复用同一入口。
- [x] 5.3 升级 `scripts/debug_eval_consistency.py`，按 root/run/checkpoint/patterns 输出 JSON 与 Markdown 一致性报告。
- [x] 5.4 确认 ordinary proto 三 seed 与 BTAPA tau1 三 seed 配置满足当前禁用项、seed 和输出路径约束。
- [x] 5.5 新增 `scripts/run_proto_vs_btapa_8gpu.sh`，支持 8 卡单进程单 GPU、skip/only/dry-run/auto-resume/skip-completed/eval-after-train。
- [x] 5.6 修 `scripts/reevaluate_apples_to_apples.py` 使用统一 resolver 与 missing pattern，并支持 fresh eval 缺失 checkpoint warning。
- [x] 5.7 新增 `scripts/analyze_proto_vs_btapa_seeds.py`，读取 fresh eval 输出 mean±std、delta 和保守 observation。
- [x] 5.8 修 `scripts/summarize_missing_runs.py` 输出并行日志、exit code、best epoch/checkpoint、final epoch 和 expected epochs 字段。
- [x] 5.9 运行 dry-run/help/focused tests 与 OpenSpec strict validate；不启动完整训练、不覆盖已有结果。
