## 1. 输入盘点与配置骨架

- [x] 1.1 在实现开始时打印并保存关键文件清单：label mapping 文件、旧 prediction 文件、旧 metrics/summary 文件、DBA 实现位置、MMW 数据 CSV 与 split metadata 文件。
- [x] 1.2 新增 `configs/mmw_town_gps_adapter_v2.yaml`，声明 `mapping_enabled` 默认配置、`mapping_disabled` 切换、四个 scene、split、model、loss、adapt、metrics 和 ablation 矩阵。
- [x] 1.3 在配置解析中接入已有 `resolve_beam_label_mapping()`，确保 mapping_enabled 复用现有 calibration artifact，mapping_disabled 使用 raw label space。
- [x] 1.4 准备新增/修改文件清单并在实现 PR/总结中列出，预计包含 `src/kd_sensing/evaluation/metrics.py`、`src/kd_sensing/losses/circular.py`、`src/kd_sensing/models/mmw_town_gps_v2.py`、`src/kd_sensing/engine/mmw_town_gps_v2.py`、`src/kd_sensing/cli/mmw_town_gps_v2.py`、`src/kd_sensing/cli/plot_mmw_town_gps_v2.py`、`src/kd_sensing/cli/compare_mmw_town_gps_v2.py`、`pyproject.toml`、README 和相关 tests。

## 2. Circular Metrics 与 Loss

- [x] 2.1 扩展 `src/kd_sensing/evaluation/metrics.py`，提供 `circular_beam_distance`、`circular_topk_min_distance`、exact/pm1/pm2/pm4、mean/median circular error、DBA 和 DBA=0 ratio helper。
- [x] 2.2 确保现有 `calculate_dba_score()` DBA 公式继续使用 circular distance，并支持 v2 summary 所需 top-L/zero-ratio 输出。
- [x] 2.3 新增 circular soft target loss 模块，支持 `circular_soft_target`、`circular_soft_ce_loss` 和 `focal_circular_soft_ce_loss`。
- [x] 2.4 实现 class-balanced weighting：`none`、`inverse_freq`、`inverse_sqrt_freq`、`effective_num`，并记录 fit split、histogram 和 weight metadata。
- [x] 2.5 添加 metrics/loss 单元测试，覆盖 0/63 wrap-around、torch/numpy/scalar 输入、top-k min distance、DBA=0 ratio、soft target normalization 和 class weight 计算。

## 3. 数据特征、Support Split 与 Branch

- [x] 3.1 实现 MMW Town GPS v2 样本加载，优先使用 `dataset/MMW/sunny/Prepared/<scene>/splits/l5p3_group_safe/` 下 CSV 和 split metadata。
- [x] 3.2 构造 BS/RSU-centric GPS 几何特征 `[E_norm, N_norm, sin_theta, cos_theta, log_range_norm, sin_heading, cos_heading, speed_norm]`，并禁止 raw lat/lon 进入模型 tensor。
- [x] 3.3 实现只从 train/source split fit 的 scaler，保存 scaler metadata，并处理 heading/speed 缺失 fallback 与 coverage 记录。
- [x] 3.4 实现 `temporal_first`、`random` 和 `trajectory` support selection，支持 `support_ratio` 与 `support_num`，写出 support manifest。
- [x] 3.5 实现 branch id 解析与 pseudo branch 生成：优先使用 `branch_id`/`trajectory_id`，否则用 numpy deterministic k-means/silhouette，样本不足时 k=1。
- [x] 3.6 添加数据/branch 测试，验证 label-space 切换、support/query 不交叉、feature scaler split 来源和 pseudo branch fallback。

## 4. GPS Backbone 与 SceneAdapterV2

- [x] 4.1 新增 MMW Town GPS v2 MLP backbone，默认 hidden_dim=128、dropout=0.1、GELU，输出 64-beam residual logits。
- [x] 4.2 保留并接入 v1 baseline adapter，不删除现有 GPS/window/coarse anchor baseline。
- [x] 4.3 实现 SceneAdapterV2 `circular_affine`，包含 scene-specific `psi`、`delta`、`scale`、`sigma`、`tau` 和 forward/reverse mixture。
- [x] 4.4 实现 SceneAdapterV2 `circular_affine_spline`，包含 periodic residual bins、linear interpolation 和 circular smoothness regularization。
- [x] 4.5 实现 SceneAdapterV2 `branch_mixture_circular`，支持 hard branch 参数选择、branch support fallback 和 metadata。
- [x] 4.6 实现 ablation 路径：`backbone_only`、`adapter_v1`、`circular_affine`、`circular_affine_spline`、`branch_mixture_circular`、`branch_mixture_circular_weighted`、`geo_only`、`geo_plus_backbone`。
- [x] 4.7 添加模型测试，覆盖 forward shape、wrap-around center、probability mixture、spline smoothness、branch fallback、residual_scale 初始化和非法 scene_id 错误。

## 5. Runner 与 Few-shot Adaptation

- [x] 5.1 新增包内 runner `kd_sensing.engine.mmw_town_gps_v2`，执行 `source_other_three`、`target_adapt_beambench` 和 `within_scene_train`。
- [x] 5.2 实现 source_other_three 训练/评估：每次留一个 scene 为 target，用其它三个 scene 训练，target label 只用于最终评估。
- [x] 5.3 实现 target_adapt_beambench：加载 source backbone，冻结 GPSBackbone，为 target scene 初始化 adapter，执行 grid search + support 梯度优化。
- [x] 5.4 实现 within_scene_train 同场景上界，并在 metadata/summary 中标记为 sanity/upper-bound protocol。
- [x] 5.5 写出标准 artifact：`summary_overall.csv`、`summary_by_scene.csv`、`predictions.csv`、`residual_by_theta_bin.csv`、`residual_by_branch.csv`、`run_metadata.json` 和配置快照。
- [x] 5.6 添加 runner smoke 测试，使用小样本或 synthetic fixture 验证三类 protocol、所有 summary 字段和 target support/query 防泄漏。

## 6. CLI、可视化、旧诊断对比与 README

- [x] 6.1 新增 `kd_sensing.cli.mmw_town_gps_v2` 并在 `pyproject.toml` 注册 `kd-sensing-mmw-town-gps-v2`。
- [x] 6.2 新增 plotter CLI，注册 `kd-sensing-plot-mmw-town-gps-v2`，生成 ENU scatter、prediction scatter、error heatmap、signed residual vs theta、residual histogram、branch visualization 和 label distribution 对比图。
- [x] 6.3 新增 comparison CLI，注册 `kd-sensing-compare-mmw-town-gps-v2`，输出 `comparison_with_previous.csv` 和 `comparison_report.md`。
- [x] 6.4 更新 README 的 “MMW Town GPS-only v2: circular scene adapter” 小节，说明失败原因、circular distance、label-space、adapter 版本、命令、summary 解读、crossroad/Hroad 残差和多模态非目标边界。
- [x] 6.5 添加 CLI help 与 artifact schema 测试，确保三个 console scripts 的 `--help` 可在 `conda run -n kd_mm_beam` 中运行。

## 7. 验证与回归

- [x] 7.1 运行 `openspec validate add-mmw-town-gps-adapter-v2 --strict` 并修复所有问题。
- [x] 7.2 运行 `openspec status --change add-mmw-town-gps-adapter-v2`，确认 proposal、design、specs 和 tasks 均完成。
- [x] 7.3 运行 `conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 --config configs/mmw_town_gps_adapter_v2.yaml --label-space mapping_enabled`。
- [x] 7.4 运行 `conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 --config configs/mmw_town_gps_adapter_v2.yaml --label-space mapping_disabled`。
- [x] 7.5 运行 `conda run -n kd_mm_beam kd-sensing-plot-mmw-town-gps-v2 --results-dir outputs/analysis/mmw_town_gps_adapter_v2/mapping_enabled`。
- [x] 7.6 运行 `conda run -n kd_mm_beam kd-sensing-compare-mmw-town-gps-v2 --previous-dir outputs/analysis/mmw_town_label_distribution --new-dir outputs/analysis/mmw_town_gps_adapter_v2/mapping_enabled`。
- [x] 7.7 运行相关快速测试：`conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py tests/test_circular_metrics.py -q`。
- [x] 7.8 运行架构与公共入口回归：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_cli_help.py -q`。
- [x] 7.9 视运行成本决定是否执行最终回归：`conda run -n kd_mm_beam pytest -q`，若未执行必须在总结中说明原因。
