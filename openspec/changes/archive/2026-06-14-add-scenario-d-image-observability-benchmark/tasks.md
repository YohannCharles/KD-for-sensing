## 1. Difficulty schema 与 Scenario D preset

- [x] 1.1 在 shared difficulty schema/registry 中新增 `scenario_d_image_observability` condition 解析，覆盖 `D0_full_image` 到 `D7_joint_worst_case`。
- [x] 1.2 将 D-level preset 标准化为 image observability operator 参数，并保留 weather、low-light、blur、occlusion、frame dropout、burst missing 的 sweep/default 配置。
- [x] 1.3 为未知 D-level、非法概率、非法 burst 长度和伪模态名称添加清晰配置错误。
- [x] 1.4 补充 synthetic config/schema 单元测试，使用 `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py -q` 验证解析边界。

## 2. Image observability transform

- [x] 2.1 在 `src/kd_sensing/data/difficulty/operators/` 中实现包内 `ImageObservabilityTransform` 或等价 helper，并支持用户指定的构造参数。
- [x] 2.2 实现 deterministic weather、low-light、motion blur、partial occlusion、frame dropout 和 burst missing，保持 image tensor shape/dtype 可被当前 batch 流程消费。
- [x] 2.3 写入 `image_valid_mask`、`image_dropout_mask`、`image_burst_dropout_mask`、`image_observability_score`、`image_degradation_metadata` 和 replay metadata。
- [x] 2.4 确保 corruption 与 missing 语义分离：physical corruption 默认不把整帧标为 invalid，dropout/burst missing 必须显式标记 invalid。
- [x] 2.5 添加 target/sample metadata preservation、同 seed determinism、score monotonic 和 D7 joint operator 测试。

## 3. Modality contract 与 batch metadata

- [x] 3.1 扩展 image modality difficulty metadata 字段说明，覆盖 image valid mask、observability score、dropout/burst mask、corruption type、severity、seed 和 frame range。
- [x] 3.2 扩展 batch 输入映射，使声明支持 observability-aware fusion 的模型能接收 image/GPS reliability metadata。
- [x] 3.3 保持 standard CNN+GPS、Image-AE+GPS 和现有 JEPA baseline 可忽略新增 metadata，并在 benchmark comparability metadata 中记录是否消费 reliability metadata。
- [x] 3.4 补充 batch 映射与伪模态拒绝测试，必要时运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_modality_difficulty.py -q`。

## 4. Observability-aware fusion

- [x] 4.1 新增 `src/kd_sensing/models/observability_aware_fusion.py` 或等价窄模块，实现 image/GPS latent shape 校验、reliability score 和 modality weight 计算。
- [x] 4.2 实现 adaptive fusion：对齐 projection 后输出 `z_fuse = w_img * z_img + w_gps * z_gps` 或等价 representation。
- [x] 4.3 实现 uncertainty gating，在 image observability 低于阈值且 JEPA predicted latent 可用时使用 temporal JEPA fallback。
- [x] 4.4 实现 `C3/C4 + D3/D4/D6/D7` JEPA advantage condition 标记和 diagnostics。
- [x] 4.5 添加 forward/gating 单元测试，覆盖缺失 metadata、GPS async 降权、image missing 降权、clean condition 不强制 fallback。

## 5. JEPA downstream temporal fallback

- [x] 5.1 为 JEPA downstream image encoder 增加可配置 temporal context fallback，使用 `image_history[t-4:t-1]` 或配置声明的历史窗口预测当前 latent。
- [x] 5.2 确保 fallback 不读取未来帧、不移动 target，并在 metadata 中记录 source history range、fallback 策略和受影响样本数。
- [x] 5.3 将 `image_valid_mask`、`image_observability_score` 和 benchmark condition metadata 接入 JEPA gating/fallback 决策。
- [x] 5.4 保持 mean-pooling 与 GPS-query baseline 默认行为兼容，未启用 fallback 时不改变现有配置 forward。
- [x] 5.5 添加 JEPA fallback focused tests，运行 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_jepa_gps_shortcut_benchmark.py -q` 或新增对应 focused test。

## 6. Scenario D benchmark runner 与聚合

- [x] 6.1 扩展 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` manifest schema，支持 `scenario_d_image_observability` 和 `scenario_c_x_d_image_observability` suite。
- [x] 6.2 实现 Cx-Dy 笛卡尔评估调度，记录 `gps_condition`、`image_condition`、severity、seed、difficulty digest 和 sample_count。
- [x] 6.3 校验 GPS-only、CNN+GPS、Image-AE+GPS、Image-JEPA only、Image-JEPA+GPS 模型组的 split、label space、metric profile 和 checkpoint provenance。
- [x] 6.4 聚合 Top-1、Top-3、DBA、clean delta、worst-case、RSI、phase transition、CNN vs JEPA crossing point 和 modality dominance ratio。
- [x] 6.5 在 ignored output root 下写出 `results/scenario_d_image_observability.csv`、`results/heatmap_cx_dy.npy`、`plots/robustness_surface.png`、`plots/phase_transition_curve.png` 和 `plots/modality_dominance.png`。
- [x] 6.6 补充 mock/synthetic benchmark tests，运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q`。

## 7. Config、CLI 与文档

- [x] 7.1 新增 Scenario D smoke/evaluation-only manifest 或 config preset，默认输出到 `outputs/analysis/scenario_d_image_observability/`。
- [x] 7.2 如新增包内 CLI 或 console script，同步 `pyproject.toml`、README 入口索引、`docs/project_surface_inventory.md` 脚本 allowlist 和架构边界测试。
- [x] 7.3 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md` 和 `docs/result_claims_registry.md`，标明 Scenario D 的运行状态、输出边界和 claim caveat。
- [x] 7.4 保证文档不提交真实 benchmark 结果，只记录本地产物路径、digest、状态和限制。

## 8. Validation

- [x] 8.1 运行 `openspec validate add-scenario-d-image-observability-benchmark --strict`。
- [x] 8.2 运行 focused tests：`conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py tests/test_jepa_gps_shortcut_benchmark.py -q`。
- [x] 8.3 涉及 CLI/config 后运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 8.4 涉及模型/fusion 后运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_cls_token_transformer_fusion.py tests/test_gps_conditioned_jepa.py -q`。
- [x] 8.5 在最终说明中记录未运行的长耗时真实 benchmark、数据/checkpoint 前置条件和剩余风险。
