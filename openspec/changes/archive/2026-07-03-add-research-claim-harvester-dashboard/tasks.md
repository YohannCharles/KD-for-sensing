## 1. Harvester Schema 与读取器

- [x] 1.1 定义 claim candidate、comparability warning、ledger record 和 dashboard summary schema。
- [x] 1.2 实现 Scene31 missing-pattern CSV/JSON 读取器，覆盖 run name、method、seed、pattern 和核心指标抽取。
- [x] 1.3 实现训练 run artifact 读取器，抽取 `final_config.yaml`、`metrics.csv/json`、`train_log.json`、`run_status.json` 和 checkpoint sidecar。
- [x] 1.4 添加 synthetic fixture tests，不读取真实 `dataset/`。

## 2. Comparability 与 Ledger

- [x] 2.1 实现 strict comparability gate，检查 split、sample_count、label_space、metric_profile、target_source、difficulty_digest 和 seed。
- [x] 2.2 实现 JSONL ledger writer，默认输出到 `outputs/analysis/research_ledger/`。
- [x] 2.3 支持 ledger CSV 导出；SQLite 后端可作为可选实现或明确 deferral。
- [x] 2.4 添加 provenance incomplete、not_comparable、needs_review 的 focused tests。

## 3. Dashboard 与 Run Index 接入

- [x] 3.1 扩展 run index adapter，暴露 harvester 所需 run identity、eval artifact 和 active process 字段。
- [x] 3.2 新增 dashboard CLI 或脚本，输出人类可读摘要和 JSON。
- [x] 3.3 dashboard 聚合 active OpenSpec、run state、GPU/进程、pending claim、candidate 和 next-action hint。
- [x] 3.4 确认 dashboard 只读，不启动训练、不清理、不改文档。

## 4. 文档与验证

- [x] 4.1 更新 README、`docs/experiment_matrix.md` 或主线文档，说明 harvester/dashboard/ledger 的用途和草稿边界。
- [x] 4.2 运行 `openspec validate add-research-claim-harvester-dashboard --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest` 的相关 focused tests，例如 run index、harvester、dashboard fixtures。
- [x] 4.4 检查 `git status --short --untracked-files=all`，确认未纳入 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或真实 metrics。
