## Why

当前仓库已经积累了大量 Scene31 missing-pattern 评估、训练日志、checkpoint sidecar 和主线文档，但从本地运行产物到可引用 claim 仍主要依赖人工查找和手工抄表。这个 change 先补研究速度最高收益的一层：自动收割指标、判定 claim 升级条件、生成每日研究面板，并用轻量实验账本把运行产物和论文证据串起来。

## What Changes

- 新增只读 `research-claim-harvester` 能力，从 `outputs/`、`logs/`、eval matrix、Scene31 fresh eval、`final_config.yaml`、`metrics.csv/json`、`run_status.json` 和 checkpoint sidecar 中抽取候选 claim。
- 新增 daily dashboard CLI 或等价脚本，汇总 active change、运行状态、GPU/进程快照、pending/unverified claim、可升级 claim、失败/缺失 run 和下一步建议。
- 新增轻量 experiment ledger，默认写入 ignored `outputs/analysis/research_ledger/`，可选导出 JSONL/SQLite/CSV，记录 run、config digest、seed、split、metric、artifact path、claim status 和 caveat。
- 扩展 run index，使其能暴露 claim harvesting 所需的 run identity、metric path、config digest、eval artifact、checkpoint provenance 和 running/waiting/failed 状态。
- 扩展 artifact/claim 文档约束，让 harvested claim draft 不能自动改写正式 `docs/result_claims_registry.md`；只生成可审阅草稿。
- 不新增重依赖，不接入 MLflow/W&B，不移动、删除或重写任何 `outputs/`、`logs/`、cache、checkpoint 或真实数据。

## Capabilities

### New Capabilities

- `research-claim-harvester`: 覆盖指标收割、claim candidate schema、strict comparability gate、daily dashboard、experiment ledger 和只读产物边界。

### Modified Capabilities

- `experiment-run-index`: 增加 dashboard/harvester 所需的 run identity、eval artifact、config digest、active process 和 claim readiness 字段。
- `experiment-artifact-registry`: 增加 experiment ledger 与 checkpoint/run provenance 的关联要求。
- `mainline-experiment-documentation`: 增加 harvested claim draft 与正式 claim registry 的审核边界。

## Impact

- 主要影响 `src/kd_sensing/diagnostics/` 或新增窄模块、`src/kd_sensing/cli/`、`scripts/` 中的只读研究面板入口、run index 汇总 schema、相关 docs 和 tests。
- 输出默认写入 ignored `outputs/analysis/research_dashboard/` 与 `outputs/analysis/research_ledger/`；源码只提交实现、OpenSpec、测试和文档。
- 最小验证使用 `conda run -n kd_mm_beam pytest ...`；不得读取真实 `dataset/` 作为单元测试前提。
