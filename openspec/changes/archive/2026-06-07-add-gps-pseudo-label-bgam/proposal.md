## Why

论文 `https://arxiv.org/pdf/2603.15093v1` 的实验对象是 Multimodal-Wireless / MMW Town10 数据集，核心方法用历史 beam index 引导 BS/RSU 侧 LiDAR 空间注意力，并按高频通信时间线做因果对齐。本项目要与该论文比较，第一阶段实现 MUST 放在 MMW 数据集上，而不是先放在 DeepSense6G scenario31-34 上。

仓库已有 MMW Town GPS v2 circular scene adapter、MMW prepared splits、Town10 scene calibration、LiDAR/RSU frame manifest 和 BEV cache 能力。缺口是：MMW GPS logits 尚未成为 TopK candidate manifest；历史 GPS pseudo label 尚未在 MMW manifest/dataset/model 中成为一等输入；BGAM runner 还没有 MMW workflow 与 normalized gain 诊断字段。

## What Changes

- 新增 MMW-first GPS pseudo-label BGAM workflow：从 `mmw_town_gps_adapter_v2` frozen logits/probs 构建 Top8 candidate manifest、pseudo-history、GPS+LiDAR BGAM manifest 和 rerank evaluation。
- 默认数据集切换为 `dataset/MMW/sunny` 的 Town10 scenes：crossroad、skybridge、curvyroad、Hroad；主配置使用 `mapping_enabled`，`mapping_disabled` 只作为 raw-label 对照。
- pseudo-history 按 MMW trajectory 语义因果对齐，默认按 `scene + agent + split` 分组，避免不同车辆或轨迹串历史。
- LiDAR 默认使用 MMW frame manifest 中的 RSU LiDAR path，支持 BEV cache 缺失时重建；manifest 记录 LiDAR source、cache path、availability 和 missing reason。
- evaluation/predictions/summary 继续报告 TopK、DBA 和 circular error，同时为 MMW 增加 GPS/final normalized gain 与 delta vs GPS，便于对照论文指标。

## Capabilities

### New Capabilities
- `gps-pseudo-label-bgam`: 定义历史 GPS pseudo label 的生成、时间对齐、mapped label-space 契约、BGAM 消费方式、防泄漏要求和评估产物。
- `mmw-town-gps-top8-candidate-selector`: 从 MMW GPS v2 logits 构建 mapped Top8 candidate manifest。
- `mmw-town-gps-lidar-bgam-reranker`: 在 MMW Town10 上运行 GPS pseudo-history + RSU LiDAR BGAM reranker。

### Modified Capabilities
- `gps-coarse-anchor-prediction`: 将 GPS coarse/pseudo 输出扩展为可导出的历史 pseudo label 序列，供 MMW BGAM 和后续 residual/fusion workflow 消费。

## Impact

- 主要代码影响：`configs/mmw_town_top8_selector.yaml`、`configs/mmw_town_gps_lidar_bgam.yaml`、MMW TopK/BGAM manifest builder、MMW BGAM CLI/runner wrapper、通用 `GPSLidarBGAMDataset`、`GPSGuidedBGAM`/`GPSLidarBGAMBeamPredictor`、BGAM evaluator 和 README。
- 数据契约影响：MMW manifest 新增 `history_pseudo_beams`、`history_pseudo_probs`、`history_pseudo_entropy`、`history_timestamps`、`history_alignment_policy`、scene-specific `beam_label_mapping_fingerprint`、`lidar_source`、`beam_power_path` 和 normalized-gain 字段。
- 实验影响：主结果目录切到 `outputs/analysis/mmw_town_top8_selector/mapping_enabled/` 和 `outputs/analysis/mmw_town_gps_lidar_bgam/mapping_enabled/`；raw-label 结果不得与 mapped 主结果混合聚合。
- 依赖影响：不引入 LLM/GPT 依赖；实现复用现有 PyTorch、NumPy、CSV/manifest、MMW GPS v2 artifacts、LiDAR BEV 和 beam label calibration。
