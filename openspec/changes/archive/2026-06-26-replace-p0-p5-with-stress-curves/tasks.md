## 1. 契约和配置入口

- [x] 1.1 将 predictive canonical preset 从旧 P0-P5 默认条件改为 clean anchor + `image_missing`、`image_noise`、`gps_noise` stress curves。
- [x] 1.2 为 legacy P0-P5 preset 增加显式兼容路径或 deprecated 标记，确保默认主评估不再生成低区分度 P-level。
- [x] 1.3 更新 diagnostic manifest/config 示例，声明 stress suite、severity values、severity unit、seed、split 和输出目录。

## 2. Difficulty pipeline 和扰动语义

- [x] 2.1 实现或复用 `image_missing` sweep：固定 tensor shape，按 severity 缺失末端连续帧，写入 `image_valid_mask=false` 和 `image_observability_score=0`。
- [x] 2.2 实现或复用 `image_noise` sweep：只启用一个视觉干扰轴，记录 degradation type、severity、作用帧范围和 replay seed。
- [x] 2.3 实现或复用 `gps_noise` sweep：只启用一个 GPS 干扰轴，记录 perturbation mode、severity、mask/delay/counterfactual fields 和 replay seed。
- [x] 2.4 增加可选 `joint_stress` 显式路径，并确保输出标记为 diagnostic 而非 primary claim。

## 3. 指标聚合和 claim gate

- [x] 3.1 更新 predictive benchmark normalization，使 stress rows 包含 model、suite、condition、severity、severity_unit、seed、split、sample_count、primary metric、clean delta 和 retention。
- [x] 3.2 实现 curve summary：`S@drop<=0.02`、`S@drop<=0.05`、`AUC_retention`、`collapse_s` 和 `weakest_axis`。
- [x] 3.3 更新 claim gate，使主 claim 使用 stress summary margin-vs-ResNet，不再使用 P0-P5 mean。
- [x] 3.4 缺少 clean anchor、strict comparable rows 或 clean anchor 过低时，将 summary/claim 标记为 unavailable、not-comparable 或 clean_anchor_unstable。

## 4. 测试

- [x] 4.1 使用 `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py -q` 覆盖 stress preset normalization、旧 P-level 降级和三条单轴扰动 mask 语义。
- [x] 4.2 使用 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q` 覆盖 benchmark manifest normalization、stress summary 和 claim gate。
- [x] 4.3 使用 synthetic batch 测试同一 suite/severity/seed/sample id 的扰动 determinism 和 target/sample metadata 不变。

## 5. 文档和验证

- [x] 5.1 更新 README 或 docs 中 predictive robustness 说明，明确主评估为 stress curves，旧 P0-P5 仅为 legacy/deprecated。
- [x] 5.2 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md` 和 `docs/result_claims_registry.md` 中的主指标与 claim provenance。
- [x] 5.3 运行 `openspec validate replace-p0-p5-with-stress-curves --strict`。
- [x] 5.4 按触碰范围运行 focused tests；若触碰架构边界或 CLI，再运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_cli_help.py -q`。
