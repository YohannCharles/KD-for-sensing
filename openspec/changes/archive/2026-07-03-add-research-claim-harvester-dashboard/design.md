## Context

仓库已经有 `kd-sensing-runs`、`docs/result_claims_registry.md`、Scene31 missing-pattern CSV/JSON、fresh eval summary 和 checkpoint sidecar，但它们之间还没有一个只读的证据流水线。研究者每天需要手动判断哪些 run 完成、哪些指标可比、哪些 claim 仍 pending、哪些结果能升级为论文表格候选。

本设计把研究证据层分为三件小工具：harvester 负责从产物收割结构化候选，dashboard 负责给人看当天状态，ledger 负责保留机器可读历史。三者都保持只读扫描，输出写入 ignored `outputs/analysis/`。

## Goals / Non-Goals

**Goals:**

- 自动发现训练、评估和 Scene31 missing-pattern/fresh-eval 产物。
- 生成 claim candidate、strict comparability warning、delta vs baseline 和 next-action hint。
- 生成轻量 experiment ledger，避免每次从日志和目录重新推断。
- 给出 daily dashboard，聚合 active OpenSpec、运行状态、GPU/进程、pending claim 和可升级 claim。
- 保持正式 claim registry 人工审核，不自动改写论文结论。

**Non-Goals:**

- 不接入 MLflow、W&B、数据库服务或外部平台。
- 不移动、删除、压缩或重写任何 `outputs/`、`logs/`、checkpoint、cache 或真实数据。
- 不自动把 harvested candidate 写入 `docs/result_claims_registry.md`。
- 不解决缺失模态统计显著性和 stress benchmark 的具体计算；这些由 `add-missing-modality-statistics-stress-suite` 处理。

## Decisions

1. **harvester 复用 run index，但不塞进 run index。**
   `kd-sensing-runs` 继续负责只读发现 run、状态和资源；claim harvester 在其输出之上读取 metrics/eval artifacts。这样 run index 不变成 claim 规则库。

2. **ledger 默认 JSONL，SQLite 作为可选后端。**
   JSONL 最小、可 diff、无需依赖，适合先落地；SQLite 可作为后续实现，用于本地查询大量 run。两者 schema 保持同字段。

3. **claim candidate 是草稿，不是 claim。**
   harvester 输出 `candidate_status`、`claim_readiness` 和 `required_review`。只有人工更新 `docs/result_claims_registry.md` 后才算正式 claim。

4. **dashboard 只读聚合，不负责调度训练。**
   dashboard 可以提示“缺 seed3”或“checkpoint 不可比”，但不启动训练、不杀进程、不清理产物。

5. **优先支持现有高价值产物。**
   首批输入覆盖 `metrics.csv/json`、`train_log.json`、`final_config.yaml`、`run_status.json`、checkpoint sidecar、`*_missing_patterns.csv/json`、Scene31 fresh eval summary 和 `kd-sensing-runs` JSON。

## Risks / Trade-offs

- **Risk:** 不同实验家族字段名不一致，harvester 误判可比性。  
  **Mitigation:** 先实现 schema-normalizer 和 required-field warning；字段缺失时只输出 `not_comparable` 或 `needs_review`。

- **Risk:** dashboard 输出太多，反而噪声更大。  
  **Mitigation:** 默认只显示 active/running/failed/pending/upgradable 摘要，完整明细写 JSON。

- **Risk:** ledger 与本地文件状态漂移。  
  **Mitigation:** 每条 ledger 记录包含 artifact path、mtime、size、可选 digest 和 generated_at；后续重新 harvest 可标记 stale。

- **Risk:** 用户误把 candidate 当正式结果。  
  **Mitigation:** 所有候选表必须包含 `candidate_only=true` 或 `claim_status=draft`，paper export 默认拒绝未审阅 candidate。

## Migration Plan

1. 增加 harvester schema、artifact reader 和 synthetic fixture。
2. 扩展 run index JSON 字段或提供 harvester-side adapter。
3. 增加 `research_dashboard` CLI/脚本和 JSON/Markdown 输出。
4. 增加 JSONL ledger writer 与 append/update 规则。
5. 更新主线文档，说明 candidate 到 claim registry 的人工审核流程。
6. 回滚时删除新增 CLI/diagnostics 模块和 OpenSpec delta；已有运行产物不受影响。

## Open Questions

- SQLite 是否首版就实现，还是只在 JSONL 数据量过大后再加。
- dashboard 是否需要读取 `openspec list --json`，还是由用户手动传入 active change 状态。
- claim candidate 的 baseline reference 是否首版只支持 Scene31 uniform winner，还是读取文档中的 claim registry 作为通用 baseline。
