## ADDED Requirements

### Requirement: MMW prepared manifests are loadable by modality-aware datasets
数据构建流程 MUST 能识别 MMW 准备流程生成的 manifest/CSV，并在配置选择 `data.dataset.type: mmw` 与 `data.dataset.scene: town10_skybridge_seed24` 时构建对应 dataset。启用模态推导、按需读取、beam 历史标签和 future beam 目标标签的语义 MUST 与现有 beam 预测流程保持一致。

#### Scenario: MMW mmWave-only 按需读取
- **WHEN** 用户使用 MMW manifest 运行 `experiment.task: mmwave`
- **THEN** dataset MUST 只读取历史 `mmwave*` power vector、`beam*` 和 `future_beam*` 标签文件
- **AND** dataset MUST 不读取 image、LiDAR、GPS 或 RSU radar 文件
- **AND** 返回样本 MUST 包含 `mmwave`、`input_beam` 和 `target_beam`

#### Scenario: MMW image+mmWave fusion 按需读取
- **WHEN** 用户使用 MMW manifest 运行 fusion 配置且启用 `["image", "mmwave"]`
- **THEN** dataset MUST 读取历史前向 RGB image、历史 mmWave power vector、历史 beam 和 future beam 标签
- **AND** dataset MUST 不要求未启用的 LiDAR、GPS 或 RSU radar 文件存在
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: MMW dataset returns stable beam and modality tensors
MMW dataset MUST 返回与现有训练流程兼容的 `input_beam` 和 `target_beam` 张量。启用 MMW 派生 mmWave 输入时，`mmwave` MUST 为 `[seq_len, 64]` 的 `torch.float32` 张量；启用 image、LiDAR 或 GPS 时，对应字段 MUST 使用现有 batch 准备流程可消费的稳定 shape 和 dtype。

#### Scenario: MMW beam 标签 shape 稳定
- **WHEN** MMW dataset 配置 `seq_len=8` 且 `num_pred=3`
- **THEN** 单样本 `input_beam` MUST 为长度 8 的整数张量
- **AND** 单样本 `target_beam` MUST 为长度 3 的整数张量
- **AND** batch 后 `target_beam` MUST 保持 `[batch_size, 3]`

#### Scenario: MMW mmWave 张量 shape 稳定
- **WHEN** MMW dataset 启用 mmWave modality
- **THEN** 单样本 `mmwave` MUST 为 `torch.float32`
- **AND** `mmwave` shape MUST 为 `[seq_len, 64]`
- **AND** 每个时隙 MUST 与同一行 CSV 的 `beam*` 历史标签时隙对齐
