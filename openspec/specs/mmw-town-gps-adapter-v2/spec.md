# mmw-town-gps-adapter-v2 Specification

## Purpose
定义 MMW Town GPS-only v2 adapter 的 label-space contract、circular metrics、SceneAdapterV2 变体、GPS 几何特征、few-shot target adaptation、ablation matrix 和结果诊断产物，用于可审计地比较 raw/mapped 64-beam 场景适配效果。
## Requirements
### Requirement: MMW Town GPS-only v2 label-space contract
系统 MUST 为 MMW Town GPS-only v2 提供显式 label-space 选择。默认 label space MUST 为 `mapping_enabled`，并 MUST 通过已有 MMW beam label calibration helper 解析 scene-specific mapping；`mapping_disabled` MUST 使用 raw 64-beam label space 作为对照。所有结果 artifact MUST 记录 `label_space`、`beam_label_space`、mapping 参数和 mapping fingerprint。

#### Scenario: 默认使用 mapping_enabled
- **WHEN** 用户运行 MMW Town GPS-only v2 runner 且未传入 `--label-space`
- **THEN** 系统 MUST 使用 `mapping_enabled`
- **AND** 系统 MUST 从配置或已有分析目录读取 mapping_enabled mapping artifact
- **AND** 输出 summary 和 predictions MUST 记录 mapping fingerprint

#### Scenario: 可切换到 mapping_disabled
- **WHEN** 用户运行 MMW Town GPS-only v2 runner 并传入 `--label-space mapping_disabled`
- **THEN** 系统 MUST 使用 raw 64-beam label
- **AND** 输出目录 MUST 与 `mapping_enabled` 分离
- **AND** comparison 或 aggregation MUST NOT 混合不同 label space 的指标

### Requirement: Circular beam metrics
系统 MUST 为 MMW Town GPS-only v2 使用 circular beam distance 作为主误差定义。指标 helper MUST 支持 numpy array、torch tensor 和标量输入，并 MUST 至少提供 circular distance、top-k circular min distance、exact、pm1、pm2、pm4、mean circular error、median circular error、DBA 和 DBA=0 ratio。

#### Scenario: 0 和 63 相邻
- **WHEN** `num_beams=64` 且系统计算 beam `0` 与 beam `63` 的距离
- **THEN** circular beam distance MUST 等于 `1`
- **AND** mean/median circular error、pmN accuracy 和 DBA MUST 使用该 circular distance

#### Scenario: 主表输出完整 circular 指标
- **WHEN** runner 写出 `summary_overall.csv` 或 `summary_by_scene.csv`
- **THEN** 表中 MUST 包含 `DBA`、`DBA_zero_ratio`、`mean_circular_error`、`median_circular_error`、`exact_acc`、`pm1_acc`、`pm2_acc`、`pm4_acc`、`top1`、`top3` 和 `top5` 中适用于该粒度的字段

### Requirement: SceneAdapterV2 variants
系统 MUST 新增 SceneAdapterV2，并 MUST 保留 SceneAdapterV1 作为可运行 baseline。SceneAdapterV2 MUST 支持 `circular_affine`、`circular_affine_spline` 和 `branch_mixture_circular` 三类 adapter。

#### Scenario: circular_affine 生成几何 logits
- **WHEN** adapter 类型为 `circular_affine`
- **THEN** 每个 scene MUST 维护 orientation offset、beam shift、beam-angle scale、geometry width、temperature 和 forward/reverse mixture 参数
- **AND** forward/reverse 分布 MUST 先归一化为概率再混合
- **AND** 返回的 `geo_logits` MUST 为 `[B, num_beams]`

#### Scenario: circular_affine_spline 使用周期 residual
- **WHEN** adapter 类型为 `circular_affine_spline`
- **THEN** 系统 MUST 在 affine center 上加入 angle-conditioned residual table
- **AND** residual table 的首尾 bin MUST 按周期邻接计算 smoothness regularization

#### Scenario: branch_mixture_circular 使用 hard branch
- **WHEN** adapter 类型为 `branch_mixture_circular`
- **THEN** 系统 MUST 优先使用样本已有 `branch_id` 或 `trajectory_id`
- **AND** 若字段缺失，系统 MUST 按 scene 生成 deterministic pseudo branch
- **AND** 每个 branch MUST 选择一套 SceneAdapterV2 参数，第一版 MUST NOT 要求 gating network

### Requirement: GPS v2 feature and logits composition
系统 MUST 为 MMW Town GPS-only v2 使用 BS/RSU-centric 局部几何特征，不得直接把 raw latitude/longitude 作为主模型输入。默认 feature 向量 MUST 包含 ENU、theta、range、heading 和 speed 的归一化或三角函数表示，并 MUST 只从 train/source split 估计标准化参数。

#### Scenario: 构造 GPS v2 feature
- **WHEN** 系统从 MMW Town CSV 或 manifest 构造 GPS v2 样本
- **THEN** 输入特征 MUST 包含 `E_norm`、`N_norm`、`sin_theta`、`cos_theta`、`log_range_norm`、`sin_heading`、`cos_heading` 和 `speed_norm`
- **AND** scaler metadata MUST 记录 fit split 和样本数
- **AND** raw latitude/longitude MUST NOT 作为模型输入特征写入 tensor

#### Scenario: geo plus residual logits
- **WHEN** ablation 为 `geo_plus_backbone`
- **THEN** 系统 MUST 计算 `final_logits = geo_logits + residual_scale * residual_logits`
- **AND** `residual_scale` MUST 初始化为配置值，默认 `0.1`

### Requirement: MMW Town GPS v2 ablation matrix
系统 MUST 支持 MMW Town GPS-only v2 的可配置 ablation 矩阵。矩阵 MUST 至少包含 `backbone_only`、`adapter_v1`、`circular_affine`、`circular_affine_spline`、`branch_mixture_circular`、`branch_mixture_circular_weighted`、`geo_only` 和 `geo_plus_backbone`。

#### Scenario: summary 按 ablation 分组
- **WHEN** runner 完成一个 label space 的实验
- **THEN** `summary_overall.csv` 和 `summary_by_scene.csv` MUST 包含 `ablation` 字段
- **AND** 每个启用 ablation MUST 能被单独过滤和比较

### Requirement: Few-shot target adapter adaptation
系统 MUST 支持 MMW Town GPS-only v2 的 few-shot target adapter adaptation。adaptation MUST 加载 source_other_three 训练得到的 backbone，默认冻结 GPSBackbone，为 target scene 初始化新的 SceneAdapterV2，并只用 target support set 优化允许的 adapter 参数。

#### Scenario: grid search 初始化 target adapter
- **WHEN** target adaptation 启用 grid search
- **THEN** 系统 MUST 在 support set 上搜索 `psi`、`delta`、`scale` 和 forward/reverse flip
- **AND** 选择标准 MUST 为 mean circular distance 或 circular soft CE
- **AND** target query/test label MUST NOT 参与初始化选择

#### Scenario: branch support 不足时退化
- **WHEN** branch adapter 中某个 branch 的 support 样本数小于 `min_branch_support`
- **THEN** 该 branch MUST 退化为全局 adapter 参数或记录不可单独适配原因
- **AND** metadata MUST 记录 fallback branch 和样本数

### Requirement: V2 result artifacts and diagnostics
系统 MUST 为 MMW Town GPS-only v2 写出机器可读结果、残差诊断、图形诊断和旧诊断对比产物。默认输出根目录 MUST 为 `outputs/analysis/mmw_town_gps_adapter_v2/<label_space>/`。

#### Scenario: runner 写出标准表
- **WHEN** MMW Town GPS-only v2 runner 完成
- **THEN** 输出目录 MUST 包含 `summary_overall.csv`、`summary_by_scene.csv`、`predictions.csv`、`residual_by_theta_bin.csv`、`residual_by_branch.csv`、`run_metadata.json` 和配置快照

#### Scenario: plotter 写出结构残差图
- **WHEN** plotter 读取 v2 results dir
- **THEN** 系统 MUST 为每个 scene 写出真实 label scatter、预测 label scatter、circular error heatmap、signed residual vs theta、residual histogram 和 label distribution 对比图
- **AND** crossroad 和 Hroad MUST 额外写出 branch_id 可视化或不可用原因

#### Scenario: comparison 输出旧诊断对比
- **WHEN** comparison 工具读取旧诊断目录和 v2 结果目录
- **THEN** 系统 MUST 写出 `comparison_with_previous.csv` 和 `comparison_report.md`
- **AND** report MUST 说明 crossroad、Hroad、curvyroad 和 skybridge 相对旧结果的变化
