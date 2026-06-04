## Context

当前 MMW Town GPS-only 诊断已经覆盖四个 sunny/Town10 场景：`crossroad`、`skybridge`、`curvyroad`、`Hroad`。诊断结论显示 within-scene train 接近上界，source_other_three 直接跨场景弱，target_adapt_beambench 能显著改善 skybridge/curvyroad，但 crossroad/Hroad 仍存在结构性 residual 和 label imbalance。这个问题更像 label topology、scene orientation/calibration、分支轨迹和残差建模问题，而不是需要更复杂多模态模型的问题。

已盘点到的关键复用点：

- label mapping 文件：`outputs/analysis/mmw_town_label_distribution/mapping_enabled/beam_label_mapping_midpoint_31_32.json`、`outputs/analysis/mmw_town_label_distribution/target_adapt_beambench_dba_calibration.json`、`source_other_three_calibration.json`、`within_scene_train_calibration.json`、`target_adapt_beambench_calibration.json`。
- prediction/metrics 文件：`outputs/gps_coarse_anchor/source_other_three_auto_mapping_dba/*/predictions.csv`、`outputs/gps_coarse_anchor/target_adapt_beambench_dba/*/predictions.csv`、`outputs/gps_coarse_anchor/within_scene_upper_bound_all_train/*/predictions.csv` 及其 `metrics.json`/`summary.json`。
- label distribution 和 residual 诊断：`outputs/analysis/mmw_town_label_distribution/<label_space>/label_distribution/*summary.json`、`gps_prediction_trajectory/*/*summary.json`、`prediction_error_label_distribution/*summary.json`。
- DBA 实现：`src/kd_sensing/evaluation/metrics.py` 中 `calculate_dba_score()` 已使用 `_circular_class_distance()`，可以作为 v2 DBA 的公式来源继续扩展，不需要重写不一致的 DBA。
- mapping helper：`src/kd_sensing/data/beam_label_calibration.py` 已提供 `resolve_beam_label_mapping()`、raw/calibrated forward/inverse、distribution reorder 和 fingerprint。
- GPS 几何 helper：`src/kd_sensing/data/geometry_residual.py` 与 `src/kd_sensing/baselines/gps_window/geometry.py` 已提供 circular distance、angle-to-beam 和 beam score kernel，可作为 v2 几何概率初始化的基础。
- MMW 数据 CSV/metadata：四个场景下存在 `dataset/MMW/sunny/Prepared/<scene>/manifests/frame_manifest.csv`、`splits/l5p3_group_safe/{train,test,all_sequences}.csv` 和 `split_metadata.json`，可以优先使用 group-safe split。

仓库架构要求新入口位于 `src/kd_sensing` 包内，不能新增 `python -m src.run_*` 这类绕过包结构的根入口。因此用户示例命令会被实现为等价包内 CLI/console scripts，例如 `kd-sensing-mmw-town-gps-v2`、`kd-sensing-plot-mmw-town-gps-v2` 和 `kd-sensing-compare-mmw-town-gps-v2`，README 给出 `conda run -n kd_mm_beam ...` 形式。

## Goals / Non-Goals

**Goals:**

- 交付一个显式 opt-in 的 MMW Town GPS-only v2 工作流，能完整跑 `mapping_enabled` 和 `mapping_disabled`。
- 保留 v1 GPS/window/coarse anchor baseline，新增 v2 backbone + SceneAdapterV2，不破坏既有 GPS teacher/student 和 HiST-Beam。
- 所有主评估表使用 circular beam distance，并输出 DBA、DBA=0 ratio、mean/median circular error、exact/pm1/pm2/pm4、top1/top3/top5。
- few-shot target adaptation 只读取 target support label，target query/test 只用于最终评估和离线诊断。
- 产物要能回归到已有诊断结果，尤其解释 crossroad/Hroad 是否由 global shift、spline residual 或 branch residual 改善。

**Non-Goals:**

- 不加入 camera、LiDAR、radar、mmWave sensing feature 或 path/radio oracle。
- 不修改 MMW raw channel/codebook 计算，不重写已有 beam label calibration helper。
- 不用 target_test/query label 做 mapping 拟合、grid search、early stopping、temperature fitting 或 branch 选择。
- 不引入 scikit-learn 等新重依赖；branch pseudo clustering 用 numpy 实现小型 deterministic k-means/silhouette，或在样本不足时退化为单分支。
- 不声明 v2 一定提升所有 ablation；若新结果变差，comparison report 必须记录具体 ablation 和可能原因。

## Decisions

1. **v2 入口作为包内实验 workflow，而不是根目录脚本**

   新增 `kd_sensing.cli.mmw_town_gps_v2`、`kd_sensing.cli.plot_mmw_town_gps_v2`、`kd_sensing.cli.compare_mmw_town_gps_v2`，并在 `pyproject.toml` 声明 console scripts。核心实现放在 `kd_sensing.engine`、`kd_sensing.models`、`kd_sensing.losses` 或 `kd_sensing.evaluation` 的窄模块中。

   替代方案是按需求文本新增 `src/run_mmw_town_gps_v2.py`。这会绕过当前包结构和项目架构约束，也会使导入、测试和 console script 管理变散，因此不采用。

2. **label_space 默认 mapping_enabled，但保持 mapping_disabled 同等可运行**

   `configs/mmw_town_gps_adapter_v2.yaml` 默认 `data.label_space: mapping_enabled`。mapping_enabled 读取 `outputs/analysis/mmw_town_label_distribution/target_adapt_beambench_dba_calibration.json` 或配置指定 mapping file，并通过 `resolve_beam_label_mapping()` 生成 scene-specific mapping；mapping_disabled 使用 raw label space。所有输出目录按 `<label_space>` 分开，summary 和 predictions 必须记录 `beam_label_space` 和 mapping fingerprint。

   替代方案是直接把 CSV label 覆盖为 mapped label。这样会丢掉 raw provenance，也容易混淆旧 checkpoint 与新 run，因此不采用。

3. **主指标统一为 circular metrics，DBA 复用现有公式并扩展 summary**

   在 `kd_sensing.evaluation.metrics` 暴露 `circular_beam_distance()`、`circular_topk_min_distance()`、`beam_classification_circular_summary()` 等 helper。现有 `calculate_dba_score()` 已使用 circular class distance；v2 只扩展 top-L、DBA=0 ratio、mean/median circular error 和 pmN accuracy 输出，不改变既有 DBA 默认语义。

   替代方案是 v2 runner 私有实现 metrics。这样会导致 DBA、0/63 邻接和 summary 字段难以与训练/评估入口共享，因此不采用。

4. **GPS 特征使用局部几何，不把 raw lat/lon 输入模型**

   v2 样本特征为 `[E_norm, N_norm, sin_theta, cos_theta, log_range_norm, sin_heading, cos_heading, speed_norm]`。`E/N/log_range/speed` 标准化只从当前 train/source split 估计，并保存 scaler metadata。缺失 heading/speed 时使用可审计 fallback：heading 可由相邻 GPS/pose 或 0 角默认值推导，speed 缺失时置 0 并记录 mask/coverage。

   替代方案是复用 raw GPS 序列或 lat/lon 输入 MLP。它弱化几何可解释性，也与当前诊断中基于 BS/RSU-centric angle 的结论不一致，因此不采用。

5. **SceneAdapterV2 先建可解释几何概率，再叠加轻量 residual**

   `circular_affine` 按 scene 学习 `psi_s`、`delta_s`、`log_scale_s`、`log_sigma_s`、`log_tau_s` 和 `flip_logit_s`。它根据 `theta` 得到 forward/reverse center，使用 circular distance 形成 `p_f`/`p_r`，先 softmax 成概率，再按 `alpha=sigmoid(flip_logit_s)` 混合概率并取 `log(p_geo+eps)`。

   `circular_affine_spline` 在 affine center 上增加 scene-specific periodic residual table，默认 `num_bins=16`，按 theta circular bin 线性插值，并用周期相邻差分 smoothness regularization。

   `branch_mixture_circular` 为每个 scene/branch 维护一套 affine+spline 参数。若 CSV 有 `branch_id` 或 `trajectory_id` 则优先使用；否则按 scene 对 `[E, N, sin_heading, cos_heading, log_range]` 做 deterministic k-means，k 候选按配置选择，样本不足时 k=1。第一版使用 hard branch，避免小样本下训练 gating network 过拟合。

6. **Backbone 只学习 residual，geo prior 和 residual 可做 ablation**

   GPS backbone 是两层 hidden MLP，默认 hidden_dim=128、dropout=0.1、GELU，输出 residual logits。最终默认 `final_logits = geo_logits + residual_scale * residual_logits`，`residual_scale` 初始化 0.1 且可训练。ablation 覆盖 `backbone_only`、`adapter_v1`、`circular_affine`、`circular_affine_spline`、`branch_mixture_circular`、`branch_mixture_circular_weighted`、`geo_only` 和 `geo_plus_backbone`。

   替代方案是直接训练一个更深 MLP 预测 64 类。它不能解释 crossroad/Hroad 的系统 shift 与 branch residual，也更难判断 few-shot 收益来自哪里，因此不作为主路径。

7. **few-shot target adaptation 使用 grid init + adapter-only optimization**

   对每个 target scene，先用其它三个 scene 训练 source backbone。target adaptation 冻结 GPSBackbone，初始化 target SceneAdapterV2：`psi` 默认 73 点、`delta` 按 beam 步长、scale 候选 `[0.75,1.0,1.25]`、flip 取 forward/reverse，用 support set 的 mean circular distance 或 circular soft CE 选最优。之后只优化 adapter 参数和允许的 regularization 项，默认不训练 residual_scale。

   branch 模式先全局 affine 初始化，再按 branch 估计 delta residual；若某 branch support 数小于 `min_branch_support`，退化到全局 adapter。support 默认 `temporal_first`，同时支持 `random` 和 `trajectory`。

8. **loss 是 supervised beam smoothing，不是 KD**

   新增 circular soft target 与 circular soft CE/focal loss。class weight 从当前训练 split label histogram 计算，支持 `none`、`inverse_freq`、`inverse_sqrt_freq`、`effective_num`。日志字段使用 `loss/beam_circular_soft_ce`、`loss/beam_focal_circular_soft_ce` 或等价非 KD 命名，不生成 distillation 字段。

9. **输出分为结果、残差诊断、图和旧诊断对比四层**

   Runner 写入 `summary_overall.csv`、`summary_by_scene.csv`、`predictions.csv`、`residual_by_theta_bin.csv`、`residual_by_branch.csv`、`run_metadata.json` 和 `resolved_config.yaml`。Plotter 只读这些结果并写入 `figures/`。Comparison 工具读取旧 `outputs/analysis/mmw_town_label_distribution` 和新结果，写 `comparison_with_previous.csv` 与 `comparison_report.md`，报告 crossroad/Hroad 改善、curvyroad/skybridge 是否保持、以及收益来源。

## Risks / Trade-offs

- [Risk] v2 结果可能不优于 target_adapt_beambench 旧诊断 → Mitigation：comparison report 必须逐 scene/ablation 记录退化，并保留 v1 baseline。
- [Risk] branch pseudo clustering 在小样本上不稳定 → Mitigation：k 候选受 scene 配置限制，样本不足退化 k=1，输出 branch count 和 fallback reason。
- [Risk] mapping_enabled 与 mapping_disabled 混算指标 → Mitigation：输出目录、metadata、summary key 和 comparison 均按 label_space 分组，mapping fingerprint 不匹配时拒绝合并。
- [Risk] target support/query 泄漏 → Mitigation：support selection manifest 记录 sample_id、split、mode、seed；adapter optimizer 和 grid search 禁止读取 query/test label。
- [Risk] 新 CLI 入口与用户示例命令不完全同名 → Mitigation：README 提供包内等价命令，并说明项目不维护 `src.run_*` 根入口。
- [Risk] circular loss class weighting 放大少数类噪声 → Mitigation：默认 `class_weight: none`，weighted ablation 单独输出并与 unweighted 分开汇总。

## Migration Plan

1. 先实现 reusable metrics/loss/model/data helpers 与单元测试，确保 `mapping_enabled`/`mapping_disabled` 都可跑小样本 smoke。
2. 接入 v2 runner，先跑 `geo_only`、`backbone_only` 和 `circular_affine_spline` 小矩阵，确认 summary/predictions schema 稳定。
3. 接入 branch、weighted loss、plotter 和 comparison 工具，补齐 README。
4. 完整执行 mapping_enabled 与 mapping_disabled 命令。若需要回滚，删除或停用新增 v2 配置/console script 即可；既有 v1/GPS/HiST-Beam 路径不受影响。

## Open Questions

- mapping_enabled 的默认 mapping file 是否固定为 `target_adapt_beambench_dba_calibration.json`，还是应由配置显式传入并在缺失时回退到 `beam_label_mapping_midpoint_31_32.json`？
- Hroad/crossroad 的 `trajectory_id` 是否已在所有 split CSV 中稳定存在；若没有，pseudo branch 是否应以 agent 分组作为优先 fallback？
- v2 完整矩阵是否默认训练所有 8 个 ablation，还是默认跑核心 4 个并通过 CLI override 扩展完整 sweep？
