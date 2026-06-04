## ADDED Requirements

### Requirement: DeepSense6G GPS+LiDAR BGAM reranker workflow
系统 MUST 提供显式 opt-in 的 DeepSense6G GPS+LiDAR BGAM reranker workflow。该 workflow MUST 默认覆盖 scenario31、scenario32、scenario33 和 scenario34，使用 `mapping_disabled`、`num_beams=64`、GPS v2 r15 logits/Top8 candidates、LiDAR BEV 或 raw point cloud，并将 GPS v2 作为 frozen spatial/candidate prior。

#### Scenario: 默认 BGAM 配置
- **WHEN** 用户运行默认 GPS+LiDAR BGAM 配置
- **THEN** 系统 MUST 解析场景为 scenario31-34
- **AND** 系统 MUST 使用 64-beam circular label 语义
- **AND** 系统 MUST 默认读取 GPS v2 r15 Top8 manifest 或先运行 BGAM manifest enrich
- **AND** 系统 MUST 将输出写入 `outputs/analysis/deepsense6g_gps_lidar_bgam/r15/mapping_disabled/`

#### Scenario: 主方法为 GPS prior 约束下的 LiDAR rerank
- **WHEN** 系统训练或评估默认 BGAM 主方法
- **THEN** final prediction MUST 默认来自 GPS Top8 candidate beams
- **AND** LiDAR MUST 只作为 GPS prior 约束下的空间证据、candidate score 或 rerank evidence
- **AND** 系统 MUST NOT 把 image/RGB/camera encoder 作为该 workflow 的输入或依赖

### Requirement: GPS/RSU geometry prior
系统 MUST 提供可复用 GPS/RSU geometry utility，用于将用户坐标转换为 RSU frame AoD。utility MUST 支持 local x/y 与 lat/lon 输入，输出 `theta_gps` radians、distance 和坐标来源 metadata。

#### Scenario: local 坐标转换 AoD
- **WHEN** 输入包含 `user_x,user_y,rsu_x,rsu_y,rsu_yaw`
- **THEN** 系统 MUST 计算 `dx=user_x-rsu_x` 和 `dy=user_y-rsu_y`
- **AND** 系统 MUST 计算 `theta_gps_global=atan2(dy,dx)`
- **AND** 系统 MUST 输出 `theta_gps=wrap_to_pi(theta_gps_global-rsu_yaw)`
- **AND** 输出 angle MUST 使用 radians

#### Scenario: lat/lon 坐标转换 AoD
- **WHEN** 输入包含 `user_lat,user_lon,rsu_lat,rsu_lon,rsu_yaw`
- **THEN** 系统 MUST 先以 RSU 为原点转换为 local ENU 或 equirectangular x/y
- **AND** 系统 MUST 使用与 local 坐标相同的公式计算 RSU frame AoD
- **AND** metadata MUST 记录转换方法为 `enu`、`equirectangular` 或等价标识

#### Scenario: wrap_to_pi 边界
- **WHEN** 输入 angle 超出 `[-pi, pi]`
- **THEN** `wrap_to_pi` MUST 返回位于 `[-pi, pi]` 或实现定义等价闭开区间内的角度
- **AND** `pi` 与 `-pi` 附近的 wrap-around MUST 在单元测试中覆盖

### Requirement: BGAM manifest enrich
系统 MUST 提供 BGAM manifest enrich 能力，从现有 Top8 candidate manifest、DeepSense6G sequence CSV 或用户提供 manifest 中补齐 GPS AoD prior、LiDAR path/cache path、RSU pose、distance、beam angle convention 和 availability 字段。enrich 后 manifest MUST 可审计并可复现。

#### Scenario: 从 Top8 manifest 补齐 BGAM 字段
- **WHEN** 用户运行 BGAM manifest enrich 且输入为 Top8 candidate manifest
- **THEN** 系统 MUST 保留 candidate beams/probs/logits、GPS context、support/query role、target label 和 Top8 hit/miss 字段
- **AND** 系统 MUST 新增或确认 `theta_gps`、`distance_to_rsu`、`lidar_path` 或 `lidar_bev_cache_path`、`rsu_yaw`、`coordinate_frame` 和 `beam_angle_convention`
- **AND** enrich metadata MUST 记录输入 manifest、原始 CSV 或 join key 来源

#### Scenario: 必要字段缺失时可诊断失败
- **WHEN** 启用 LiDAR BGAM 但 manifest 无法解析 LiDAR path 或 BEV cache path
- **THEN** 系统 MUST 早失败或按配置跳过 LiDAR/BGAM ablation
- **AND** 错误或 skipped reason MUST 包含缺失字段名和受影响样本数

### Requirement: GPSLidarBGAMDataset
系统 MUST 提供 GPS+LiDAR BGAM dataset，用于读取 BGAM manifest 并返回稳定 batch 字段。dataset MUST 支持 variable-size raw point cloud collate、BEV cache tensor、GPS prior、candidate metadata 和 anti-leakage diagnostics。

#### Scenario: dataset 返回 BGAM 样本字段
- **WHEN** dataset 读取一个 BGAM manifest 样本
- **THEN** 返回样本 MUST 包含 `candidate_beams: LongTensor [K]`
- **AND** 返回样本 MUST 包含 `candidate_probs` 或 `candidate_log_probs`
- **AND** 返回样本 MUST 包含 `theta_gps: FloatTensor []` 和 `distance_to_rsu: FloatTensor []`
- **AND** 返回样本 MUST 包含 `gt_beam` 或 `target_label` 仅作为 loss/evaluation label
- **AND** 返回样本 MUST 包含 `sample_id`、`scene` 和 support/query role metadata

#### Scenario: LiDAR BEV cache 样本
- **WHEN** manifest 行包含有效 `lidar_bev_cache_path`
- **THEN** dataset MUST 按样本懒加载该 BEV tensor
- **AND** 返回 `lidar_bev` MUST 为 `torch.float32` 且形状为 `[C,H,W]`
- **AND** dataset 初始化 MUST NOT 全量读取 BEV cache 目录

#### Scenario: raw point cloud 样本
- **WHEN** manifest 行包含 raw `lidar_path` 且配置使用 raw pillar profile
- **THEN** dataset MUST 读取点云为 `[N,3]` 或 `[N,4]`
- **AND** collate_fn MUST 安全批处理 variable-size point clouds
- **AND** 空点云 MUST 生成固定尺寸全零 BEV 或全零 pillar pseudo-image

### Requirement: LiDAR spatial encoder
系统 MUST 提供轻量 LiDAR spatial encoder，用于输出供 BGAM 和 cross-attention 使用的 `[B,C,H,W]` BEV feature。默认 encoder MUST 消费现有 3-channel BEV；fallback `SimplePillarEncoder` MUST 支持 raw point cloud 到 pillar pseudo-image 的纯 PyTorch/NumPy 实现。

#### Scenario: 现有 BEV 输入输出空间特征
- **WHEN** encoder 接收 `lidar_bev: [B,C0,H,W]`
- **THEN** encoder MUST 输出 `bev_feat: [B,C,H',W']`
- **AND** 输出 MUST 保留二维空间维度供 BGAM mask 对齐
- **AND** encoder 参数量和输出通道 MUST 可配置

#### Scenario: SimplePillarEncoder 构造 pseudo-image
- **WHEN** fallback encoder 接收 raw point cloud batch
- **THEN** 系统 MUST 按配置 ROI 和 grid size 过滤点并分配 BEV cell
- **AND** 每个 cell 的基础特征 MUST 至少覆盖 point count normalized、mean z、max z、mean intensity、mean x offset 和 mean y offset
- **AND** 输出 pseudo-image MUST 可被小型 2D CNN 消费

#### Scenario: freeze_lidar_encoder 配置
- **WHEN** 配置设置 `model.lidar.freeze_lidar_encoder=true`
- **THEN** LiDAR spatial encoder 参数 MUST 不参与 optimizer 更新
- **AND** run metadata MUST 记录该 encoder frozen

### Requirement: GPSGuidedBGAM mask/gate
系统 MUST 提供 `GPSGuidedBGAM` module，对 LiDAR BEV feature 应用 GPS-derived angular mask 或 soft gate。BGAM MUST 将 BEV cell center angle 作为 buffer 保存，并支持 `single_soft`、`single_hard`、`topk_union_soft` 和 `topk_per_candidate` 模式。

#### Scenario: single_soft mask
- **WHEN** BGAM mode 为 `single_soft`
- **THEN** 系统 MUST 计算 `delta=wrap_to_pi(theta_cell-theta_gps)`
- **AND** mask MUST 等于或等价于 `exp(-0.5*(delta/sigma)^2)`
- **AND** 输出 mask shape MUST 为 `[B,1,H,W]`
- **AND** soft mask 最大值 MUST 位于最接近 `theta_gps` 的 BEV cell 附近

#### Scenario: single_hard mask
- **WHEN** BGAM mode 为 `single_hard`
- **THEN** mask MUST 在 `abs(delta)<=half_width` 的 cell 上为 1
- **AND** 其它 cell MUST 为 0
- **AND** mask MUST 为 binary tensor 或等价的 0/1 float tensor

#### Scenario: topk_union_soft mask
- **WHEN** BGAM mode 为 `topk_union_soft` 且输入包含 `gps_topk_beams` 与 `gps_topk_probs`
- **THEN** 系统 MUST 将 candidate beam 转换为 angle
- **AND** 每个 candidate MUST 生成 soft angular mask
- **AND** union mask MUST 使用 weighted max、max 或配置指定等价聚合
- **AND** 输出 mask shape MUST 为 `[B,1,H,W]`

#### Scenario: topk_per_candidate mask
- **WHEN** BGAM mode 为 `topk_per_candidate`
- **THEN** 系统 MUST 返回 candidate-specific masked BEV features
- **AND** 输出 shape MUST 为 `[B,K,C,H,W]`
- **AND** 每个 candidate mask MUST 只由对应 candidate angle/prob 和 GPS prior uncertainty 生成

### Requirement: beam angle table
系统 MUST 支持 beam index 到 angle 的可配置映射。系统 MUST 优先读取配置提供的 `beam_angle_table.npy` 或等价表；缺失时 MAY 使用 DFT-ULA approximation，但 MUST 在 metadata 中记录 approximation 和 convention。

#### Scenario: 使用配置 beam angle table
- **WHEN** 配置提供 beam angle table 且 shape 与 `num_beams` 匹配
- **THEN** 系统 MUST 使用该表将 beam index 映射为 radians angle
- **AND** 系统 MUST 校验表中 angle 为 finite number

#### Scenario: 使用 DFT-ULA fallback
- **WHEN** 未提供 beam angle table
- **THEN** 系统 MUST 使用配置声明的 fallback convention 计算 beam angle
- **AND** fallback metadata MUST 记录 `beam_angle_source=dft_ula_approximation`
- **AND** comparison report MUST 标明 beam-angle convention 可能影响 BGAM 解释

### Requirement: LiDAR BEV cross-attention
系统 MUST 提供 `LidarBEVCrossAttention` module，对 BGAM 后的 BEV spatial tokens 做 learnable-query cross-attention，并输出固定维 LiDAR embedding。

#### Scenario: cross-attention 输入输出形状
- **WHEN** 输入 `masked_bev_feat` shape 为 `[B,C,H,W]`
- **THEN** 系统 MUST flatten spatial dimensions 为 `[B,H*W,C]` tokens
- **AND** learnable query MUST attend to BEV tokens as key/value
- **AND** 输出 `lidar_emb` MUST 为 `[B,d_model]`

#### Scenario: 多 query pooling
- **WHEN** 配置 `num_queries>1`
- **THEN** 系统 MUST 使用多个 learnable queries
- **AND** query outputs MUST 被 mean pooling、attention pooling 或配置指定方式聚合为单个 LiDAR embedding

### Requirement: GPS prior encoder
系统 MUST 提供 `GPSPriorEncoder`，将 GPS-derived AoD prior 和可选 GPS logits/probs 编码为固定维 GPS embedding。该 encoder MUST 是轻量 MLP，并 MUST 不训练或修改外部 GPS v2 model。

#### Scenario: 最小 GPS prior 特征
- **WHEN** 输入包含 `theta_gps` 与 `distance_to_rsu`
- **THEN** encoder MUST 至少使用 `sin(theta_gps)`、`cos(theta_gps)` 和 `log(distance+eps)`
- **AND** 输出 `gps_emb` MUST 为 `[B,d_model]`

#### Scenario: 使用 GPS logits/probs 特征
- **WHEN** batch 包含 `gps_logits` 或 `gps_probs`
- **THEN** encoder MAY 投影 64 维 logits/probs 或其 TopK summary
- **AND** encoder MAY 使用 GPS entropy 和 top1 margin
- **AND** run metadata MUST 记录 GPS logits/probs source 为 frozen prior

### Requirement: GPSLidarBGAMBeamPredictor
系统 MUST 提供 GPS+LiDAR BGAM beam predictor，用于组合 GPS prior encoder、LiDAR spatial encoder、BGAM、LiDAR cross-attention 和 fusion/rerank heads。默认 forward MUST 返回 TopK rerank scores，并 MAY 返回 full 64 logits 作为兼容或 ablation。

#### Scenario: forward 返回 TopK rerank 输出
- **WHEN** batch 包含 LiDAR 输入、GPS prior 和 TopK candidate metadata
- **THEN** 模型 MUST 输出 `final_candidate_scores: [B,K]`
- **AND** 模型 MUST 输出 `candidate_probs: [B,K]`
- **AND** 模型 MUST 输出 `selected_beam: [B]`
- **AND** 默认 final score MUST 等于 `candidate_log_prob + lambda_lidar * lidar_or_fused_score`

#### Scenario: fusion 模式可配置
- **WHEN** 配置 `model.fusion=cross_attention`
- **THEN** learnable query MUST attend to GPS embedding 和 LiDAR embedding tokens
- **AND** 输出 fused embedding MUST 用于 candidate scoring
- **WHEN** 配置 `model.fusion=concat_mlp`
- **THEN** 系统 MUST 使用 concat GPS/LiDAR embedding 的 MLP 作为 fallback

#### Scenario: full 64 head 兼容输出
- **WHEN** 配置启用 `model.full64_head.enabled=true`
- **THEN** 模型 MAY 输出 `logits64: [B,64]`
- **AND** 该输出 MUST NOT 取代默认 TopK rerank 主报告

### Requirement: BGAM training losses
系统 MUST 支持 GPS+LiDAR BGAM 训练 loss。默认 loss MUST 包含 TopK candidate CE 或 candidate soft CE；可选 full 64 CE、label smoothing 和 prior anchor loss MUST 可配置。

#### Scenario: TopK rerank CE
- **WHEN** ground truth beam 位于 GPS TopK candidates 中
- **THEN** 系统 MUST 使用 `target_candidate_index` 计算 candidate CE
- **AND** loss metadata MUST 记录参与 rerank CE 的样本数

#### Scenario: TopK miss 样本
- **WHEN** ground truth beam 不在 GPS TopK candidates 中
- **THEN** 系统 MUST 跳过该样本的 target-candidate-index CE
- **AND** 系统 MAY 使用 nearest-candidate soft loss 或 full 64 CE
- **AND** loss metadata MUST 记录 skipped rerank samples

#### Scenario: circular label smoothing
- **WHEN** 配置启用 circular/local label smoothing
- **THEN** soft target MUST 按 64-beam circular distance 构造
- **AND** wrap-around beam MUST 被正确处理

### Requirement: BGAM training and evaluation protocol
系统 MUST 支持 GPS+LiDAR BGAM 的训练、评估和 ablation matrix。GPS v2 logits/probs MUST frozen；target query MUST 只用于最终 evaluation、predictions、figures 和 comparison report。

#### Scenario: 默认 ablation matrix
- **WHEN** 用户运行默认 BGAM workflow
- **THEN** summary MUST 至少包含 `gps_only`、`lidar_only_no_bgam`、`gps_lidar_no_bgam`、`gps_lidar_soft_bgam`、`gps_lidar_hard_bgam`、`gps_lidar_topk_union_bgam` 和 `gps_lidar_topk_per_candidate_rerank`
- **AND** 每个 ablation MUST 记录启用的 GPS/LiDAR inputs、BGAM mode 和 beam angle source

#### Scenario: checkpoint 选择
- **WHEN** workflow 训练可学习 BGAM/rerank 模型
- **THEN** checkpoint MUST 按配置选择 val/top1 或 val/DBA 最优
- **AND** target query label MUST NOT 用于 checkpoint selection

#### Scenario: GPS-only baseline
- **WHEN** 系统评估 `gps_only`
- **THEN** prediction MUST 来自 frozen GPS v2 logits/probs 或 Top8 candidate prob
- **AND** 该 baseline MUST 与 GPS v2 r15 report 使用相同 circular metrics 口径

### Requirement: BGAM evaluation artifacts
系统 MUST 写出 GPS+LiDAR BGAM 的 metrics、predictions、debug mask、ablation summary 和 run metadata。所有 beam error MUST 使用 64-beam circular distance，所有主结果 MUST 与 GPS-only baseline 比较。

#### Scenario: metrics 与 summary 输出
- **WHEN** BGAM evaluation 完成
- **THEN** 系统 MUST 写出 `metrics.json`
- **AND** 系统 MUST 写出 `summary_overall.csv`、`summary_by_scene.csv` 和 `summary_by_bgam_mode.csv`
- **AND** summary MUST 包含 Top1、Top3、Top5、Top8、DBA、mean/median circular error、delta vs GPS 和 sample count

#### Scenario: predictions 字段
- **WHEN** 系统写出 `predictions.csv`
- **THEN** 每行 MUST 包含 `sample_id`、`scene`、`gt_beam`、`gps_top1`、`pred_beam`、`gps_topk`、`model_topk`、`correct`、`target_in_topk`、`bgam_mode`、`beam_angle_source` 和 `scenario_id` 或 scene 等价字段

#### Scenario: debug mask 输出
- **WHEN** 配置 `debug_masks.enabled=true`
- **THEN** 系统 MUST 为抽样样本写出 BGAM mask PNG 或 tensor artifact
- **AND** debug artifact MUST 记录 `theta_gps`、sigma/half_width、BGAM mode 和 sample id
- **AND** debug artifact MUST NOT 包含 ground truth beam 作为 mask source

### Requirement: BGAM anti-leakage guard
系统 MUST 明确防止 future ground-truth beam label 进入 BGAM mask、model input、normalization fit 或 target query checkpoint selection。实现 MUST 提供断言、metadata 或测试证明该约束。

#### Scenario: mask source 不包含 gt_beam
- **WHEN** 模型 forward 调用 `GPSGuidedBGAM`
- **THEN** BGAM inputs MUST 只包含 BEV feature、`theta_gps`、GPS uncertainty、GPS TopK beams/probs 或 beam angle table
- **AND** `gt_beam`、`target_label` 或 oracle candidate MUST NOT 作为 BGAM input

#### Scenario: no future label leakage 测试
- **WHEN** 开发者运行 BGAM anti-leakage 测试
- **THEN** 测试 MUST 验证改变 `gt_beam` 不改变 BGAM mask
- **AND** 测试 MUST 验证 target query rows 不参与 normalizer fit 或 checkpoint selection

### Requirement: BGAM validation and documentation
系统 MUST 提供 GPS+LiDAR BGAM 的单元测试、CPU smoke、CLI help 测试和 README 工作流说明。

#### Scenario: 单元测试覆盖核心行为
- **WHEN** 开发者运行 BGAM 相关测试
- **THEN** 测试 MUST 覆盖 `wrap_to_pi`、`gps_to_rsu_aod`、beam angle fallback、BGAM mask shape、soft mask peak、hard mask binary、TopK union mask、forward pass、backward pass 和 anti-leakage

#### Scenario: CPU smoke 可运行
- **WHEN** 开发者在无 CUDA 环境运行 BGAM smoke test
- **THEN** 系统 MUST 能使用 synthetic LiDAR BEV 或小点云完成 forward、loss、backward 和 metrics 计算

#### Scenario: README 说明运行方式
- **WHEN** README 更新完成
- **THEN** README MUST 说明 GPS+LiDAR BGAM 的输入 manifest columns、训练命令、评估命令、debug mask、结果文件、RSU coordinate assumption 和 beam-angle convention
