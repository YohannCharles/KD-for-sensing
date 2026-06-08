# 后续 attention 模块插入点

本 change 不实现新的 attention、beam-guided attention 或 cross-attention fusion，只记录最适合插入的位置。

## image encoder

候选文件：`src/kd_sensing/models/image.py`

候选类/函数：

- `ImageFeatureExtractor.forward`
- `ImageModalityNet.forward`
- `ImageStudentModalityNet.forward`

推荐插入点：

- `ImageFeatureExtractor.forward` 中 `frame_features = self.cnn_layers(frames)` 之后、flatten/fc 之前。这里仍保留二维空间 feature map，适合 image 关键区域 attention 或 beam-guided spatial gate。
- `ImageModalityNet.forward` 中 `features = self.feature_extraction(image_batch)` 之后。这里是 `[B, T, D]` 时序 feature，适合轻量 temporal/channel attention 或和 GPS/radar token 对齐。

和 BeamBench Camera AE + GPS 的关系：官方 Camera AE 先输出 camera encoder feature，再与 GPS/radar dense feature late fusion。本仓库等价位置是 image encoder 输出 `[B, T, D]` 后进入 fusion 或 GRU 之前。

## LiDAR encoder

候选文件：`src/kd_sensing/models/lidar.py`

候选类/函数：

- `LidarFeatureExtractor.forward`
- `LidarModalityNet.forward`
- `LidarStudentModalityNet.forward`

推荐插入点：

- `LidarFeatureExtractor.forward` 中 `lidar_feat = self.cnn_layers(lidar)` 后、`global_avg_pool/global_max_pool` 前。这里仍保留 BEV 空间维度 `[B*T, C, H, W]`，最适合 LiDAR 关键区域 attention 或 beam-guided spatial mask。
- `LidarModalityNet.forward` 中 `features = self.feature_extraction(lidar_batch)` 后。这里已是 `[B, T, D]` global feature，适合 temporal attention 或和 GPS pseudo-history gate 对齐，但不再保留二维空间细节。
- BGAM 相关后续优先对照 `src/kd_sensing/models/gps_lidar_bgam.py` 与 `src/kd_sensing/models/gps_lidar_bgam_model.py`，在 LiDAR feature、GPS prior/top-k candidate 和 mask/gate 生成前后插入。

## GPS embedding

候选文件：`src/kd_sensing/models/gps.py`

候选类/函数：

- `GpsFeatureExtractor.forward`
- `GpsModalityNet.forward`
- `GpsStudentModalityNet.forward`

推荐插入点：

- `GpsFeatureExtractor.forward` 中 `features = self.net(...)` 后。这里是 GPS embedding `[B, T, D]`，适合生成 beam-guided attention query、scene-conditioned gate 或 GPS prior token。
- `GpsModalityNet.forward` 中 `seq_out` 或 `enhanced_seq_out` 后。这里适合把历史 GPS/beam 的 temporal context 注入 classifier 或 fusion head。

## late fusion / concat / classifier head

候选文件：

- `src/kd_sensing/models/fusion/networks.py`
- `src/kd_sensing/models/fusion/cls_token_transformer.py`

推荐插入点：

- late concat fusion：各模态 encoder 输出 feature 后、classifier head 前。适合加 cross-modal gate、modality dropout diagnostic 或 beam-guided reweighting。
- `CLSTokenTransformerFusionNet.forward`：各模态 `features = self.encoders[modality](tensor)` 后、`torch.stack(modality_features, dim=1)` 前，可以对单模态 feature 做 attention/gate。
- `CLSTokenTransformerFusionNet.forward`：`stacked_features = torch.stack(...)` 后、`embedded_tokens = self._embed_modality_tokens(stacked_features)` 前，可以做跨模态 token attention 或 beam-guided token bias。
- `CLSTokenTransformerFusionNet.forward`：`memory = self.transformer(...)` 后、`prediction_head` 前，可以做 CLS token refinement 或 candidate rerank head。

## dataloader batch 字段

DeepSense6G 当前 batch 通常能提供：

- `image`
- `radar_ra` / `radar_da`
- `gps`
- `lidar`
- `input_beam`
- `target_beam`
- metadata 中的 scene/sample/sequence 信息，取决于 dataset 配置 `return_metadata`
- 若启用相关 manifest，可提供 timestamp、future GPS、beam power 或 candidate top-k 字段

beam-guided attention 不应在 dataloader 中提前筛掉模态或改写模型行为。数据层只负责暴露历史 beam、GPS、scene id、timestamp 和 target label；attention/gate 逻辑应留在模型或 fusion 层。
