## 1. Model And Config

- [ ] 1.1 盘点现有 image、LiDAR、radar、GPS encoder 和 fusion core，确认 AMBER-lite 第一版优先使用 `modular_sequence`。
- [ ] 1.2 新增缺失模态 fusion core 或最小 mask-token helper，并记录 `training_strategy_metadata()`。
- [ ] 1.3 新增 AMBER-lite train/eval 配置或 manifest，默认 output root 为 ignored `outputs/analysis/amber_lite_missing_modality/`。

## 2. Modality Dropout And Evaluation Conditions

- [ ] 2.1 实现训练期 modality dropout profile，支持 image、LiDAR、radar、GPS 的可配置 dropout rate、seed 和 digest。
- [ ] 2.2 确保 dropout 只改变输入和 availability/missing mask metadata，不改变 target_beam、beam_power、sample id 或 split metadata。
- [ ] 2.3 实现 AMBER-lite missing-modality evaluation suite normalization，覆盖 clean、单模态缺失、多模态缺失、poor image、LiDAR/radar unavailable 和 wrong/async GPS。

## 3. Summary And Claim Boundary

- [ ] 3.1 输出 condition-level Top-K、DBA、beam distance、missing-mask provenance 和 strict comparability fields。
- [ ] 3.2 将 AMBER-lite row 标记为 `reproduction_scope: amber_lite_local`，禁止误称完整 AMBER 官方复现。
- [ ] 3.3 缺少 real metrics、LiDAR/radar artifact 或 strict 字段时标记为 pending/unavailable/not_comparable，禁止进入 strict ranking。

## 4. Tests And Documentation

- [ ] 4.1 添加 synthetic forward tests，覆盖全模态、单模态缺失和多模态缺失。
- [ ] 4.2 添加 modality dropout policy 和 target contract focused tests。
- [ ] 4.3 添加 summary adapter fixture tests，不读取真实 `dataset/` 或 checkpoint。
- [ ] 4.4 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和必要 README 索引。

## 5. Validation

- [ ] 5.1 运行 `openspec validate reproduce-amber-lite-missing-modality-baseline --strict`。
- [ ] 5.2 运行 `conda run -n kd_mm_beam pytest <amber-lite focused tests> -q`。
- [ ] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
