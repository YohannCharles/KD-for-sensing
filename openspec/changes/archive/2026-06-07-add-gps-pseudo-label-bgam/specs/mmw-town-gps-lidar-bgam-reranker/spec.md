## ADDED Requirements

### Requirement: MMW Town GPS+LiDAR BGAM reranker workflow
系统 MUST 提供显式 opt-in 的 MMW Town GPS+LiDAR BGAM reranker workflow。该 workflow MUST 默认使用 MMW Town10 sunny scenes、`mapping_enabled`、MMW GPS v2 logits/Top8 candidates、GPS pseudo-history label 和 RSU/BS-side LiDAR BEV 或 raw point cloud，并将 GPS v2 作为 frozen spatial/candidate prior。

#### Scenario: 默认 MMW BGAM 配置
- **WHEN** 用户运行 MMW GPS+LiDAR BGAM 默认配置
- **THEN** 系统 MUST 读取 `configs/mmw_town_gps_lidar_bgam.yaml`
- **AND** 系统 MUST 使用 64-beam circular label 语义
- **AND** 系统 MUST 默认读取或生成 `outputs/analysis/mmw_town_top8_selector/mapping_enabled/manifest/top8_candidate_manifest.csv`
- **AND** 输出 MUST 写入 `outputs/analysis/mmw_town_gps_lidar_bgam/mapping_enabled/`

#### Scenario: pseudo-history 按 MMW trajectory-safe 分组
- **WHEN** 构建 MMW pseudo-history
- **THEN** 系统 MUST 默认按 `scene`、`agent` 和 `split` 分组
- **AND** 每个 history token MUST 来自当前 anchor timestamp 之前或当前时刻已可观测的信息
- **AND** 系统 MUST 输出 `history_pseudo_beams`、`history_pseudo_probs`、`history_pseudo_entropy`、`history_valid_mask`、`history_timestamps`、`history_alignment_policy` 和 missing count metadata

#### Scenario: RSU LiDAR 作为默认感知输入
- **WHEN** frame manifest 中存在 RSU LiDAR path
- **THEN** MMW BGAM manifest MUST 优先使用 RSU/BS-side LiDAR path
- **AND** manifest MUST 记录 `lidar_source=rsu`、`lidar_path`、`lidar_bev_cache_path`、`lidar_available` 和 `lidar_missing_reason`
- **AND** 若 RSU path 缺失，系统 MAY 回退到 prepared split 中的 CAV window LiDAR，并 MUST 标记 `lidar_source`

#### Scenario: BGAM 主方法只重排 GPS candidates
- **WHEN** 系统训练或评估默认 MMW BGAM 主方法
- **THEN** final prediction MUST 默认来自 GPS Top8 candidate beams
- **AND** BGAM mask/gate MUST 默认使用历史 GPS pseudo label，而不是历史真实 label
- **AND** target/query true beam label MUST NOT 进入 BGAM mask、normalizer fit、训练输入或 checkpoint selection

#### Scenario: MMW summary 输出 normalized gain
- **WHEN** MMW BGAM evaluation 完成
- **THEN** predictions SHOULD 包含 `gps_normalized_gain`、`final_normalized_gain` 和 `delta_normalized_gain_vs_GPS`
- **AND** summary SHOULD 包含 mean GPS normalized gain、mean final normalized gain 和 delta vs GPS
- **AND** normalized gain MUST 只作为 evaluation/report 指标，不得用于 pseudo-history 生成或 checkpoint selection

#### Scenario: oracle-history 只作为 upper bound
- **WHEN** 配置启用 `oracle_history_bgam_upper_bound`
- **THEN** 系统 MAY 使用历史真实 label 作为对照上界
- **AND** 输出和 summary MUST 明确标记 `uses_oracle_history_label=true`
- **AND** 该 ablation MUST NOT 作为主方法或默认 checkpoint selection 来源
