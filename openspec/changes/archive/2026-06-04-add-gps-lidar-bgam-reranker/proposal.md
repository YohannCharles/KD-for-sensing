## Why

当前 DeepSense6G GPS v2 r15 已能提供较强的 64-beam Top8 候选召回，下一步需要验证 LiDAR 几何信息是否能在不破坏 GPS prior 的前提下改善候选内选择与近邻 beam 排序。用户给出的 GPS+LiDAR BGAM 方向适合落为一个显式 opt-in 的实验 workflow：用 GPS 坐标或 GPS logits 生成 coarse AoD prior，再用该 prior 对 LiDAR BEV 空间特征做 mask/gate 和 cross-attention。

## What Changes

- 新增 GPS+LiDAR BGAM reranker workflow，默认面向 DeepSense6G scenario31-34、`mapping_disabled`、`num_beams=64` 和 GPS v2 r15 Top8 prior。
- 新增 GPS/RSU 几何工具契约：支持 local x/y 和 lat/lon 到 RSU frame AoD 的转换，提供 `wrap_to_pi`、local ENU/equirectangular 近似和 `gps_to_rsu_aod`。
- 扩展 Top8 manifest/dataset 使用方式：复用已保存 GPS v2 logits、candidate beams/probs 和 GPS context，同时补充或读取 LiDAR raw path/BEV cache path、RSU pose、AoD prior、distance 和 optional beam angle metadata。
- 新增轻量 LiDAR spatial encoder 与 BGAM 模块：支持从现有 LiDAR BEV cache 或 raw point cloud 构造 `[B, C, H, W]` BEV feature，并提供 `single_soft`、`single_hard`、`topk_union_soft` 和 `topk_per_candidate` mask/gate 模式。
- 新增 LiDAR BEV cross-attention 与 GPS prior encoder：LiDAR spatial tokens 经 BGAM 后由 learnable query 聚合，GPS prior 用小 MLP 编码 AoD、distance、GPS logits/probs、entropy 和 margin。
- 新增 GPS+LiDAR fusion/rerank 模型：默认在 GPS Top8 candidate beams 内重排，保留 `final_score_i = log_p_gps_i + lambda_lidar * lidar_score_i`；同时提供可选 sparse 64 logits 和 full 64-class head 作为兼容/ablation。
- 新增配置、包内 CLI 和 console scripts，用 `kd_sensing.cli.*` 暴露 manifest 准备、训练/评估、plot/debug mask 与 GPS v2 comparison；不新增 `train_gps_lidar_bgam.py` 或其它顶层旧入口。
- 新增 anti-leakage guard：BGAM mask 只能由 GPS 坐标、RSU pose、GPS logits/probs 和 beam angle table 生成，训练标签 `gt_beam`/future ground truth 不得进入 mask 或模型输入。
- 新增 ablation matrix：至少包含 `gps_only`、`lidar_only_no_bgam`、`gps_lidar_no_bgam`、`gps_lidar_soft_bgam`、`gps_lidar_hard_bgam`、`gps_lidar_topk_union_bgam` 和 `gps_lidar_topk_per_candidate_rerank`。
- 新增测试与文档，覆盖 geometry、mask shape/峰值、hard mask binary、forward/backward、Top8 rerank、防泄漏、CLI help 和 CPU smoke。

## Capabilities

### New Capabilities

- `deepsense6g-gps-lidar-bgam-reranker`: 定义 GPS-derived AoD prior、LiDAR BGAM mask/gate、LiDAR BEV cross-attention、GPS+LiDAR fusion/rerank、anti-leakage、训练评估输出和 ablation 契约。

### Modified Capabilities

- `project-architecture`: 新 workflow 必须落在 `src/kd_sensing/` 包内，并通过包内 CLI/console scripts 暴露，不新增顶层训练/评估脚本或兼容聚合入口。
- `modality-aware-data-loading`: Top8/BGAM dataset 必须按需读取 GPS prior 与 LiDAR raw/BEV 字段，未启用 LiDAR 或 image 时不得触发对应 IO；target query label 不得参与训练输入、mask 或 normalization。
- `lidar-preprocessing`: LiDAR BEV cache 与 raw point cloud loader 需要支持 BGAM 所需的稳定 BEV grid metadata、ROI/FoV 参数追踪和按样本懒加载。
- `experiment-workflow`: 增加 GPS+LiDAR BGAM 的配置驱动运行、标准输出文件、分层验收命令和 README 工作流说明。

## Impact

- 源码：新增或修改 `src/kd_sensing/utils/geometry.py`、`src/kd_sensing/data/`、`src/kd_sensing/data/transform_ops/lidar.py`、`src/kd_sensing/models/`、`src/kd_sensing/losses/`、`src/kd_sensing/engine/`、`src/kd_sensing/evaluation/` 和 `src/kd_sensing/cli/`。
- 配置：新增 `configs/deepsense6g_gps_lidar_bgam.yaml`，默认复用 `outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep` 与 `outputs/analysis/deepsense6g_top8_selector/r15/mapping_disabled/manifest/top8_candidate_manifest.csv`。
- 入口：新增 `kd-sensing-prepare-deepsense6g-gps-lidar-bgam-manifest`、`kd-sensing-run-deepsense6g-gps-lidar-bgam`、`kd-sensing-evaluate-deepsense6g-gps-lidar-bgam` 和可选 plot/debug CLI；所有 Python 验收命令使用 `conda run -n kd_mm_beam`。
- 测试：新增 `tests/test_gps_lidar_bgam_geometry.py`、`tests/test_gps_lidar_bgam_model.py`、`tests/test_gps_lidar_bgam_dataset.py`、`tests/test_gps_lidar_bgam_runner.py`，并扩展架构边界和 CLI help 测试。
- 产物：新增 `outputs/analysis/deepsense6g_gps_lidar_bgam/` 下的 manifest、metrics、predictions、debug masks、ablation summary、run metadata 和 comparison report；本地 outputs/logs/cache/checkpoint 仍不纳入源码变更。
