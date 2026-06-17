## 1. 归档治理收口

- [x] 1.1 更新 `docs/project_surface_inventory.md`，将 `predictive-jepa-robustness` 标记为 `current`，并说明它是 pending/unverified 的 current workflow capability，不等于真实数值 claim 已完成。
- [x] 1.2 更新 `openspec/specs/predictive-jepa-robustness/spec.md` 的 `## Purpose`，移除归档 scaffold 的 `TBD` 文案，写明 Predictive Robustness 的场景、P-suite、claim 边界和 CxD sanity 关系。
- [x] 1.3 检查 `openspec/changes/archive/2026-06-16-add-predictive-jepa-robustness/`、`openspec/specs/predictive-jepa-robustness/` 和新增 YAML 是否应随上一 change 一起提交，并在最终说明中记录它们属于归档后收口状态。

## 2. 健康护栏与 wording guard

- [x] 2.1 调整 `tests/test_architecture_boundaries.py` 的 retired-route wording guard，使其继续拒绝旧 KD、HiST/Hist、Top8 selector standalone、GPS residual、camera residual 等 active wording。
- [x] 2.2 为 current JEPA 合法语境增加测试覆盖：`GPS-query` baseline compatibility、`gps_query_pool` 对照、`condition_id_consumed=false`、`blocked_condition_fields` 和 `forbidden_condition_fields` 不应被误判为退役路线回流。
- [x] 2.3 如测试规则不足以表达语境，优先改写 current spec 文案，把 GPS-query 描述限定为现有 JEPA baseline compatibility 或对照模型，而不是旧研究线推荐入口。

## 3. Predictive Robustness claim 边界

- [x] 3.1 更新 `docs/experiment_matrix.md`、`docs/experiment_protocols.md`、`docs/mainline_model_catalog.md` 和 `docs/result_claims_registry.md`，明确单个 `P4_joint_predictive_recovery` 训练 profile 不等价于完整 P0-P5 benchmark。
- [x] 3.2 确认 predictive smoke manifest 的 synthetic metrics、mock weights、allow_missing_artifacts 和 partial/strict model group 文案始终标记为 `mock/smoke`、`pending`、`unavailable` 或 `not_comparable`。
- [x] 3.3 如需要，补充配置加载或文档同步测试，确保 predictive train config、benchmark manifest 和 claim registry 的路径/状态一致。

## 4. Benchmark runner 拆分或预算登记

- [x] 4.1 评估 `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` 中 predictive suite normalization、metric row construction、regional aggregation 和 output artifact planning 是否可安全拆到窄模块。
- [x] 4.2 若拆分风险可控，抽出 predictive helper 模块并保持 `run_jepa_gps_shortcut_benchmark` facade、result dict、CSV/JSON schema 和 output_files 注册兼容。（已评估：本 change 暂缓拆分，执行 4.3 预算路径。）
- [x] 4.3 若本 change 暂缓拆分，更新 `docs/project_surface_inventory.md` 的热点 inventory，记录当前规模、suite-specific 拆分方向、暂缓原因和后续优先级。
- [x] 4.4 为拆分或预算路径补充 focused tests，覆盖 predictive smoke manifest 的 output registration、claim status、margin-vs-CNN 和 strict comparability。

## 5. 验证

- [x] 5.1 运行 `openspec validate stabilize-predictive-jepa-governance --type change --strict`。
- [x] 5.2 运行 `openspec validate --specs --strict`。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_jepa_gps_shortcut_benchmark.py -q`。
- [x] 5.5 如 runner 拆分触碰 difficulty 或 JEPA model metadata，追加运行 `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py tests/test_gps_conditioned_jepa.py -q`。
- [x] 5.6 无法运行任何推荐验证时，在最终说明中记录原因和剩余风险。（不适用：推荐验证均已运行。）
