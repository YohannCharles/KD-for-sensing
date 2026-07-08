## ADDED Requirements

### Requirement: AMR-Net window-2 local paper alignment
AMR-Net local architecture baseline MUST remain a current `amr_net` whole-model exception，并且只使用论文模态 `image`、`lidar` 和 `gps`；这三者属于用户允许的 `image`、`radar`、`gps`、`lidar` 模态集合。 本地默认配置 MUST 使用 `seq_len=2`、`num_pred=1`，不得为了填满允许集合而新增非论文 radar branch。

#### Scenario: AMR-Net 默认窗口受限
- **WHEN** 用户加载 `configs/fusion/amr_net_supervised.yaml`
- **THEN** data 和 model 配置 MUST 声明 `seq_len=2` 与 `num_pred=1`
- **AND** `model.primary.modalities` MUST 等于 `["image", "lidar", "gps"]`
- **AND** 配置 MUST NOT 启用 `mmwave`、`csi`、历史 beam index 或旧 `amr_net_gps_image` 路线

#### Scenario: AMR-Net 不新增非论文 radar branch
- **WHEN** 构建 `amr_net`
- **THEN** 模型 MUST 只要求启用的 image、LiDAR 和 GPS batch
- **AND** 模型 MUST NOT 因用户允许 radar 而强制要求 `radar_batch`
- **AND** metadata MUST 记录该模型是 paper-aligned local baseline under allowed-modality subset
