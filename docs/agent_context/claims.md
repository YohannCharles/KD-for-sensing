# Claim 更新任务上下文

用于把本地结果、diagnostic evidence、paper export、ledger 或 reviewer-facing 表格更新到 claim registry 和主线实验文档。

## 先读

- `docs/result_claims_registry.md`
- `docs/experiment_protocols.md`
- `docs/mainline_model_catalog.md`
- `docs/mainline_experiment_history.md`
- `docs/experiment_matrix.md`
- `openspec/specs/mainline-experiment-documentation/spec.md`
- `openspec/specs/research-claim-harvester/spec.md`
- `openspec/specs/paper-artifact-export/spec.md`

## Owner

- Reviewed claims：`docs/result_claims_registry.md`
- 参数口径：`docs/experiment_protocols.md`
- 模型和 workflow 目录：`docs/mainline_model_catalog.md`
- 运行顺序和 caveat：`docs/experiment_matrix.md`
- Draft ledger、JSON summary 和 HTML readiness dashboard：ignored `outputs/analysis/`

## 边界

- 只把有 provenance、status、metric definition、split/protocol、seed/checkpoint 或 blocked reason 的结果写入 claim registry。
- `candidate_only=true`、mock、smoke、historical、upper-bound、blocked 或 pending 结果不能冒充 reviewed main claim。
- 本地 JSONL ledger、JSON/HTML dashboard、figure-data、CSV/LaTeX 草稿默认写入 ignored `outputs/`，不提交。
- HTML dashboard 只是 `candidate_only` / readiness 视图，不替代 `docs/result_claims_registry.md`，也不能把 pending、unverified 或 not_comparable candidate 写成 reviewed claim。
- claim 文档不改变训练 runtime；若 claim 需要新 workflow 或数据契约，先走 OpenSpec change。

## 验证

- `conda run -n kd_mm_beam kd-sensing-paper-export --input docs/result_claims_registry.md --output-dir outputs/paper_artifacts/current`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- 相关 workflow focused tests，按 claim 来源选择
