## 1. JEPA downstream 组件

- [x] 1.1 在 `src/kd_sensing/models/jepa_downstream.py` 新增 `hybrid_residual_query` pooler，组合 mean/content query/GPS residual query，并保持输出 `[B,T,D]`。
- [x] 1.2 扩展 JEPA downstream pooler config normalization、registry build、metadata 输出和错误信息，确保旧 `mean` 与 `gps_query_attention` 行为不变。
- [x] 1.3 为 hybrid pooler 增加 focused tests，使用 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py -q` 覆盖 shape、GPS condition 可选/必需路径、residual alpha 初始化和 diagnostics。
- [x] 1.4 扩展 `JepaContextImageEncoder` opt-in temporal auxiliary branch，暴露 current latent、temporal predicted latent、history source range、availability 和 insufficient-history metadata。
- [x] 1.5 新增 feature-consistency fusion helper 或 modular representation core，融合 current/predicted/GPS residual features，并确保不读取 `c_idx`、`d_idx`、`predictive_condition_id` 或 condition string。
- [x] 1.6 为 temporal auxiliary branch 和 feature-consistency gate 增加 tests，使用 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_modular_sequence_next_query_transformer.py -q` 验证无未来信息、默认行为兼容和 diagnostics 输出。

## 2. Predictive Robustness difficulty pipeline

- [x] 2.1 在 shared difficulty preset/operator 层新增 `predictive_jepa_robustness` canonical P0-P5 condition 标准化与 unknown-condition 报错。
- [x] 2.2 实现或复用 image operator 参数，支持 `P1_current_frame_missing_history_available`、`P2_semantic_occlusion_history_available`、`P4_joint_predictive_recovery` 和 `P5_novel_weather_history_available`。
- [x] 2.3 实现 plausible wrong GPS 扰动，记录 source sample、scene constraint、distance/beam offset criteria、seed、fallback 和 counterfactual status。
- [x] 2.4 确保 predictive operators 输出 valid masks、observability score、source indices、history availability 和 replay metadata，且不移动 target、beam power、sample id 或 split metadata。
- [x] 2.5 增加 difficulty tests，使用 `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py -q` 覆盖 P-level 解析、determinism、no-label-shift、no-future-leak、plausible wrong GPS fallback 和 replay metadata。

## 3. 模型配置与训练入口

- [x] 3.1 新增 `configs/fusion/experiments/jepa_image_gps/` 下的 JEPA predictive hybrid fusion 派生配置，复用当前 BeamBench-fair/Image+GPS/JEPA checkpoint 口径。
- [x] 3.2 配置中显式启用 hybrid pooler、temporal auxiliary branch、feature-consistency gate、seq_len/history window、GPS condition dropout 或 counterfactual training profile。
- [x] 3.3 新增 predictive robustness train/eval 或 benchmark manifest 配置，默认输出到 ignored `outputs/analysis/predictive_jepa_robustness/...`。
- [x] 3.4 使用 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` 验证新增配置可加载、旧 JEPA GPS-biased/GPS-query/CNN+GPS 配置不受影响。
- [x] 3.5 如实现需要新增 representation core 或 registry component，补充 architecture boundary tests，并使用 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证入口边界。

## 4. Benchmark runner 与输出

- [x] 4.1 扩展 `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py`，解析 suite type `predictive_jepa_robustness`、P-level conditions、history window 和 output artifact plan。
- [x] 4.2 将 predictive suite 执行委托给 shared difficulty pipeline，禁止 runner 维护独立 image/GPS corruption 分支。
- [x] 4.3 扩展 model group/comparability schema，支持 `jepa_predictive_hybrid`、Image CNN+GPS 和 JEPA baseline 的 strict comparable 检查。
- [x] 4.4 扩展 condition-level CSV、regional summary 和 JSON manifest，写出 `predictive_dba`、`predictive_top1`、`cnn_predictive_dba`、`margin_vs_cnn_dba`、`claim_pass_5pt`、overall CxD sanity 和 claim status。
- [x] 4.5 增加 benchmark tests，使用 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q` 覆盖 predictive suite schema、mock/smoke claim status、strict comparability、margin-vs-CNN 和 output_files 注册。

## 5. 文档与 claim 账本

- [x] 5.1 更新 `docs/experiment_matrix.md`，加入 predictive robustness smoke、train-then-evaluate 和 real-run 命令示例，所有 Python 命令使用 `conda run -n kd_mm_beam ...`。
- [x] 5.2 更新 `docs/mainline_model_catalog.md`，登记 JEPA predictive hybrid fusion 为 pending model line，并与 GPS-biased/query-pool baseline 区分。
- [x] 5.3 更新 `docs/experiment_protocols.md`，记录 P0-P5 condition、seq_len/history window、metric profile、predictive DBA margin 和 overall CxD sanity 口径。
- [x] 5.4 更新 `docs/result_claims_registry.md`，新增 predictive robustness claim 为 pending/unverified；只有真实 strict comparable run 达到 `margin_vs_cnn_dba >= 0.05` 后才能改为 real claim。

## 6. 验证

- [x] 6.1 运行 `openspec validate add-predictive-jepa-robustness --strict`。
- [x] 6.2 运行 `openspec status --change add-predictive-jepa-robustness`，确认 artifacts apply-ready。
- [x] 6.3 运行 focused 回归：`conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_modality_difficulty.py tests/test_jepa_gps_shortcut_benchmark.py -q`。
- [x] 6.4 运行配置和架构边界回归：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_architecture_boundaries.py -q`。
- [x] 6.5 如时间允许，运行 `conda run -n kd_mm_beam pytest -q`；无法运行时在最终说明记录原因和剩余风险。
