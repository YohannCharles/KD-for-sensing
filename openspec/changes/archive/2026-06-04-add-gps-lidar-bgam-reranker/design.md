## Context

仓库当前已经完成 DeepSense6G GPS v2 adapter、TopK analysis 和 Top8 candidate selector workflow。`configs/deepsense6g_top8_selector.yaml` 默认使用 scenario31-34、`support_ratio=0.15`、`mapping_disabled` 和 64-beam circular label；`src/kd_sensing/data/deepsense6g_topk_candidate_manifest.py` 能从 GPS v2 logits 重新计算 Top8 candidates；`TopKCandidateManifestDataset` 已能返回 candidate beams/probs、GPS context 和 optional modality feature path。

LiDAR 侧已有 `kd_sensing.data.transform_ops.lidar`，支持 raw point cloud 读取、ROI/FoV 过滤、3-channel BEV 构造、cache key/hash 和懒加载；已有 `LidarFeatureExtractor`/`lidar_cnn` 主要输出序列级 embedding，不保留供 BGAM 操作的空间 BEV token。用户提出的参考论文包含 LiDAR encoder、BGAM、LiDAR cross-attention 和 cross-modality attention，但本项目不能恢复 image encoder，也不能新增顶层 `train_gps_lidar_bgam.py` 等旧入口。

因此本 change 应做成一个新的、显式 opt-in 的 GPS+LiDAR BGAM reranker：沿用 GPS v2 Top8 candidate generator 与 anti-leakage split，新增 GPS-derived AoD prior 对 LiDAR BEV 空间特征做 mask/gate，再在候选内重排 beam。

## Goals / Non-Goals

**Goals:**

- 复用现有 GPS v2 logits、Top8 candidate manifest、circular metrics、LiDAR BEV cache 和包内 CLI 结构。
- 提供 GPS/RSU geometry utility，将 local x/y 或 lat/lon 转为 RSU-frame AoD，并记录距离、yaw convention 和坐标来源。
- 提供 LiDAR spatial encoder，默认消费已有 `[C,H,W]` BEV cache，必要时支持 raw point cloud -> lightweight pillar pseudo-image fallback。
- 提供 BGAM hard/soft/topK union/topK per-candidate mask，并确保 mask 只由 GPS coordinate、RSU pose、GPS logits/probs 和 beam angle table 生成。
- 提供 LiDAR cross-attention、GPS prior encoder、GPS+LiDAR fusion 和 Top8 reranker，默认保留 GPS prior fusion。
- 提供训练/评估/ablation/report workflow，输出与 GPS v2 baseline、Top8 oracle 和 LiDAR/no-BGAM ablation 可比较的结果。
- 提供 CPU smoke test 与 no-future-label leakage 测试，保证没有 CUDA 时也能验证核心 forward/backward。

**Non-Goals:**

- 不实现 image/RGB encoder，不恢复 camera-dependent 论文路线。
- 不把 LiDAR-only 64-class classifier 作为主方法；它只作为 ablation 或 baseline。
- 不默认训练或修改 GPS v2 adapter；GPS v2 logits/probs 默认 frozen。
- 不引入 OpenPCDet、spconv 或其它重型点云依赖；第一版使用纯 PyTorch/NumPy fallback。
- 不新增顶层旧入口、`src.data.*`/`src.models.*` 聚合层或长期维护的脚本旁路。
- 不使用 future ground-truth beam label、target query label 或 oracle candidate 构造 BGAM mask。

## Decisions

### Decision 1: 以 Top8 manifest 为默认数据底座，另提供 BGAM manifest enrich step

默认输入不是重新定义一个独立 `GPSLidarDataset` CSV，而是复用 `top8_candidate_manifest.csv` 并新增 BGAM enrich step。该 step 从 DeepSense6G raw/sequence CSV 或现有 manifest 字段补齐 `lidar_path`/`lidar_bev_cache_path`、`user_x/user_y` 或 `user_lat/user_lon`、`rsu_x/rsu_y` 或 `rsu_lat/rsu_lon`、`rsu_yaw`、`theta_gps`、`distance_to_rsu` 和 LiDAR availability。

备选方案是完全新建 `datasets/gps_lidar_dataset.py` 并读任意 CSV。这里不采用作为默认路径，因为仓库已经有 Top8 manifest、support/query 防泄漏和 GPS logits 对齐逻辑；重新建一套 dataset 会增加 label leakage 和候选口径漂移风险。实现上可以提供通用 manifest column mapping，支持用户给出的字段名，但输出仍归一到包内 BGAM manifest 契约。

### Decision 2: 几何先验放在 `kd_sensing.utils.geometry`，mask 不接触 label

新增 `wrap_to_pi`、`gps_to_local_xy`、`gps_to_rsu_aod` 和 beam angle table helper。local x/y 优先；lat/lon 使用以 RSU 为原点的 equirectangular 或 ENU 近似；所有角度单位明确为 radians。RSU yaw 缺失时默认按配置选择 `0` 或早失败，并在 metadata 记录。

备选方案是在 dataset 或 BGAM module 内直接写角度转换。这里不采用，因为 AoD prior 会被 manifest builder、dataset、model 和 debug plot 同时使用，集中 utility 更容易测试坐标和 wrap-around 边界。

### Decision 3: LiDAR spatial encoder 默认消费现有 BEV cache，raw pillar encoder 是 fallback

默认路径使用现有 `build_lidar_bev`/cache 生成的 3-channel BEV，进入 `LidarBEVSpatialEncoder` 后保留 `[B,C,H,W]` 空间 feature。若配置 `lidar.profile: pillar6`，则新增纯 PyTorch/NumPy `SimplePillarEncoder` fallback：对每个 BEV cell 计算 point count、mean z、max z、mean intensity、mean x/y offset to cell center，再用小 CNN 输出空间 feature。

备选方案是引入 PointPillars/OpenPCDet。这里不采用，因为项目当前依赖轻量且 CI/smoke 需要 CPU 可跑；重依赖会明显增加安装和运行不确定性。后续若已有 OpenPCDet module 可作为 optional backend，不作为本 change 默认要求。

### Decision 4: BGAM 对 BEV feature 做 multiplicative gate，并支持 candidate-specific 输出

`GPSGuidedBGAM` 在初始化时按 BEV ROI/grid 预计算 `theta_cell [H,W]` buffer。`single_soft` 使用 GPS AoD 为中心的 Gaussian angular gate；`single_hard` 使用 half-width 二值 mask；`topk_union_soft` 使用 GPS TopK beam/prob 与 beam-angle table 生成 union gate；`topk_per_candidate` 返回 `[B,K,C,H,W]` 供 candidate-wise reranker 使用。mask 输出和 debug PNG 只表达 prior/gate，不包含 ground truth。

备选方案是使用历史最优 beam 或 gt beam 生成 BGAM，这更接近论文原设定但会在本项目中泄漏未来标签，因此明确拒绝。

### Decision 5: 主模型以 Top8 rerank 为主，64-class head 只做兼容/ablation

`GPSLidarBGAMBeamPredictor` 默认返回 Top8 candidate scores：GPS prior encoder 编码 `theta_gps`、distance 和 GPS logits/probs；LiDAR spatial encoder + BGAM + `LidarBEVCrossAttention` 输出 LiDAR embedding；fusion 可用 cross-modal attention 或 concat MLP；candidate head 以 `log_p_gps_i + lambda_lidar * score_i` 产生 final candidate scores。full 64-class CE 可作为辅助 loss 或 ablation，但主报告必须以 Top8 rerank 和 GPS baseline 比较。

备选方案是直接输出 64 类 logits 并让模型“自己学”GPS+LiDAR。这里不采用作为主方法，因为 GPS v2 已是强 candidate generator，直接 64 分类更容易掩盖 LiDAR 是否真的改善候选内选择，也更容易破坏 GPS prior。

### Decision 6: Runner 复用 Top8 selector 的输出语义，但新增 BGAM 专属 ablation

新增 engine/CLI 负责 manifest enrich、train/eval、ablation 和 comparison。输出至少包含 `metrics.json`、`summary_overall.csv`、`summary_by_scene.csv`、`summary_by_bgam_mode.csv`、`predictions.csv`、`debug_masks/`、`run_metadata.json` 和 `resolved_config.yaml`。ablation 固定包含 GPS-only、LiDAR-only no BGAM、GPS+LiDAR no BGAM、hard/soft/topK union BGAM 和 topK per-candidate rerank。

备选方案是把 BGAM 直接塞进已有 Top8 selector runner 的 optional modality 分支。这里不采用，因为 BGAM 需要空间 mask、debug mask、BEV grid metadata 和不同 loss/report 字段，单独 workflow 更清晰；但它必须复用 Top8 manifest 和 metrics helper。

### Decision 7: 训练协议沿用 support/query 防泄漏边界

默认训练模式支持 `support_only` 和 `source_pretrain_target_finetune`。GPS v2 logits frozen；target query 不参与训练、normalization fit、early stopping 或 mask 参数选择；checkpoint 按 target support internal validation 或 val split 的 DBA/top1 选择。rerank loss 对 target 不在 Top8 的样本跳过 candidate CE，但保留 full 64 CE 或 nearest-candidate soft loss（按配置）。

备选方案是用 query label 做 early stopping 来稳定小样本训练。这里不采用，因为该 workflow 的可信度依赖 target query leakage guard。

## Risks / Trade-offs

- [Risk] RSU yaw、beam angle convention 或坐标系不明导致 AoD prior 偏转。→ Mitigation: 配置必须记录 `coordinate_frame`、`yaw_unit`、`yaw_zero_axis`、`beam_angle_convention`；提供 toy geometry 测试和 debug mask/angle summary。
- [Risk] Top8 manifest 中 LiDAR path/RSU pose 字段缺失。→ Mitigation: manifest enrich CLI 早失败或按配置跳过 LiDAR ablation，并在 metadata 写明缺失字段与样本数。
- [Risk] 现有 3-channel BEV 不包含 pillar offset 等论文特征，LiDAR 增益不足。→ Mitigation: 默认先跑 cache-friendly BEV；提供 `pillar6` fallback ablation，报告 profile 和参数量。
- [Risk] BGAM hard mask 过窄导致 LiDAR evidence 被误删。→ Mitigation: 默认 `single_soft`，hard mask 仅作 ablation；sigma/half-width 支持按 GPS entropy/uncertainty 自适应。
- [Risk] TopK union beam angle table 与 dataset beam 编号不一致。→ Mitigation: 支持 config-provided table；fallback DFT-ULA approximation 仅作为明确标记的 approximation，并输出 convention warning。
- [Risk] LiDAR BEV cache 参数混用。→ Mitigation: 复用/扩展已有 cache hash metadata，run metadata 必须记录 ROI、BEV size、profile 和 cache path。
- [Risk] 模型参数和 debug mask 输出增加运行成本。→ Mitigation: 小 CNN、AMP optional、debug mask 默认只抽样保存，CPU smoke 使用极小 synthetic batch。

## Migration Plan

1. 新增 OpenSpec 契约、配置和包内 CLI，不修改既有 GPS v2、Top8 selector 或 LiDAR-only canonical 配置默认行为。
2. 先实现 geometry、beam angle table、BGAM mask 和 synthetic tests，验证不依赖真实数据。
3. 实现 BGAM manifest enrich 与 dataset/collate，确认可从 Top8 manifest 读到 GPS prior 和 LiDAR BEV。
4. 实现模型、loss、runner 和 ablation，先运行 CPU smoke，再跑小样本 DeepSense6G。
5. 接入 debug mask、summary/comparison report 和 README。
6. 如需回滚，删除新配置、CLI、engine/model/data/loss 模块和 OpenSpec change；现有 Top8 selector、GPS v2、LiDAR-only workflow 不受影响。

## Open Questions

- DeepSense6G scenario31-34 的 RSU yaw/heading 是否在 raw CSV 中稳定存在；如果不存在，默认 `rsu_yaw=0` 是否符合当前坐标系。
- 当前 Top8 manifest 的 `lidar_feature_path` 是否能回溯到 raw point cloud/BEV cache；若只能读到扁平特征，需要补充 raw sequence CSV join key。
- beam index 到物理角度的 convention 是否已有权威表；若没有，第一版只能提供 DFT-ULA fallback 并把结论标为 approximation。
- 第一版主 loss 是否默认只用 Top8 rerank CE，还是同时启用 full 64 CE 辅助头；建议默认 Top8 rerank，full 64 CE 作为 ablation。
