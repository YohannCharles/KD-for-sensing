## 1. Source Audit

- [x] 1.1 建立 WCL 2025 source-audit manifest schema，记录论文元数据、代码 URL、source commit、license、checkpoint、dataset、模态、split、metric 和 missing details。
- [x] 1.2 调研并填充官方代码/权重/训练 recipe 可用性；不可用项标记为 blocked、pending 或 unavailable。
- [x] 1.3 添加 source-audit dry-run CLI 或 owner function，默认 output root 为 ignored `outputs/analysis/wcl2025_missing_modality_reproduction/`。

## 2. Official Or Local-Substitute Branch

- [x] 2.1 若官方代码可用，实现 official-code wrapper 或 metrics/prediction ingestion，并记录 source commit 与 checkpoint provenance。
- [x] 2.2 若官方 artifact 不可用，实现 paper-aligned local substitute 配置和模型/组件路径，并记录与论文的 deviation。
- [x] 2.3 明确 official reproduction、local_substitute、blocked、pending、not_comparable 的 claim status 写出逻辑。

## 3. Missing-Modality Model And Evaluation

- [x] 3.1 优先用 `modular_sequence` 组件实现 WCL 2025 local substitute 的 per-modality encoder 和 missing-modality fusion。
- [x] 3.2 如确需 whole-model exception，补充 design note 并添加 registry build、synthetic forward、ModelOutput adaptation 和 metadata tests。
- [x] 3.3 实现 clean、单模态缺失、多模态缺失和论文关键 missing conditions 的 condition-level summary adapter。

## 4. Comparability And Documentation

- [x] 4.1 写出 split、scene set、label space、metric profile、sample count、seed、difficulty digest 和 provenance。
- [x] 4.2 strict mismatch 时标记 not_comparable/external_reference，禁止进入 strict ranking。
- [x] 4.3 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和必要 README 索引，区分 official reproduction 与 local substitute。

## 5. Tests And Validation

- [x] 5.1 添加 source-audit branch selection fixture tests，不读取真实 `dataset/`、外部 repo 或 checkpoint。
- [x] 5.2 添加 synthetic model/summary adapter focused tests。
- [x] 5.3 运行 `openspec validate reproduce-wcl2025-robust-missing-modality-baseline --strict`。
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest <wcl2025 focused tests> -q`。
- [x] 5.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
