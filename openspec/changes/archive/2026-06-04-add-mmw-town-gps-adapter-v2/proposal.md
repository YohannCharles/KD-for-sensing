## Why

MMW Town GPS-only 初步诊断显示：同场景 GPS 分类很强，但 source_other_three 直接跨场景几乎失效，target_adapt_beambench 虽能救回 skybridge/curvyroad，却在 crossroad 和 Hroad 仍留下明显结构性 circular residual。现在需要在不引入 camera/LiDAR/radar 多模态的前提下，用可解释、轻量的 GPS backbone + scene adapter v2 系统验证 circular label、scene-specific calibration、branch/residual adapter 与 label imbalance loss 是否能提升跨场景和 few-shot 适配。

## What Changes

- 新增 MMW Town GPS-only v2 实验工作流，默认使用 `mapping_enabled`，同时通过命令行支持 `mapping_disabled` 对照，并复用已有 MMW beam label calibration/mapping helper 与诊断产物。
- 新增 circular beam metrics，包括 circular distance、top-k circular min distance、exact/pm1/pm2/pm4、mean/median circular error、DBA 和 DBA=0 ratio，并要求主 summary 全部使用 circular distance。
- 新增 circular soft label loss，支持 soft target CE、focal circular soft CE，以及 `none`、`inverse_freq`、`inverse_sqrt_freq`、`effective_num` class-balanced weighting。
- 新增 GPS-only v2 模型族：轻量 GPS MLP backbone、保留 v1 baseline、SceneAdapterV2 的 `circular_affine`、`circular_affine_spline` 和 `branch_mixture_circular`，并支持 `geo_only`、`backbone_only`、`geo_plus_backbone` 等 ablation。
- 新增 MMW Town few-shot target adaptation：冻结 source backbone，为 target scene 初始化并优化 SceneAdapterV2，支持 grid search 初始化、temporal/random/trajectory support 选择和 branch-aware fallback。
- 新增配置、runner、plotter 和 previous-diagnostics comparison 工具，自动执行 `source_other_three`、`target_adapt_beambench`、`within_scene_train`，输出 summary、prediction、theta-bin、branch residual 和图形诊断产物。
- 更新 README，增加 “MMW Town GPS-only v2: circular scene adapter” 使用说明和结果解读边界。
- 不新增多模态输入，不改变现有 GPS v1 / HiST-Beam / MMW calibration 默认行为；v2 为显式 opt-in 实验路径。

## Capabilities

### New Capabilities

- `mmw-town-gps-adapter-v2`: 定义 MMW Town GPS-only v2 的模型、loss、circular metrics、few-shot adaptation、实验协议、输出 artifact、可视化和旧诊断对比契约。

### Modified Capabilities

- `gps-modality-model`: 增加显式 opt-in 的 MMW Town GPS v2 backbone/adapter 模型能力，同时保持既有 `gps_teacher`/`gps_student` 序列模型兼容。
- `soft-beam-label-training`: 扩展 circular soft target supervised loss 与 class-balanced weighting 语义，确保 v2 loss 不被记录为 KD。
- `mmw-cross-scene-adaptation-protocol`: 将 `source_other_three`、`target_adapt_beambench` 和 `within_scene_train` 作为 MMW Town GPS-only v2 可审计协议接入，并保持 target support/query 防泄漏。
- `experiment-workflow`: 增加 MMW Town GPS v2 的配置驱动 runner、plotter、comparison 命令和标准输出表/图。

## Impact

- 影响 `src/kd_sensing/evaluation/metrics.py`、loss 模块、GPS/MMW 数据特征构造、模型注册、MMW Town v2 runner/plotter/comparison CLI、配置文件、README 与测试。
- 需要读取本地 `outputs/analysis/mmw_town_label_distribution` 中已有 mapping、metrics、prediction、label distribution 诊断产物，但这些输出仍是本地运行产物，不纳入源码。
- 新增输出默认写入 `outputs/analysis/mmw_town_gps_adapter_v2/<label_space>/`，包含 CSV/JSON/figure/report，可用于复现实验和论文诊断。
- 所有项目相关 Python 验证继续使用 `conda run -n kd_mm_beam ...`。
