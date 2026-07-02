# image-only-legal-crossroad-probe Specification

## Purpose
定义 MMW Town10 crossroad 上 image-only few-shot probe 的合法输入边界、运行矩阵、feature cache 和 eligibility reporting，用于隔离图像模态能力并防止 GPS、radio/path、beam_power 或 target_test label oracle 泄漏到 adaptation。
## Requirements
### Requirement: Image-only Hist probe 已退役
Image-only legal crossroad probe 中依赖 `configs/hist_beam/`、HiST variants、V8/V9 Hist heads 或 `kd-sensing-hist-beam-loso` 的路径 MUST 从当前支持面退役。

#### Scenario: Image-only Hist probe 配置不可运行
- **WHEN** 用户引用 `configs/hist_beam/image_only_legal_crossroad_probe.yaml`
- **THEN** 系统 MUST 报告配置已退役或不存在
- **AND** 系统 MUST 不构建 image-only HiST probe model

### Requirement: Image-only probe batch 字段 allowlist
数据构建、collate 和 batch preparation MUST 在 image-only legal probe 中按启用模态和合法标签字段输出 batch。原始 manifest、CSV 或本地数据文件中存在 GPS、LiDAR、radar、mmWave、CSI、channel、path 或 beam_power 字段，不得导致这些字段进入模型输入、loss、target adaptation 或 evaluation payload。

#### Scenario: image-only batch 只暴露合法字段
- **WHEN** resolved modalities 等价于 `["image"]` 且 `protocol.image_only=true`
- **THEN** batch MUST 包含 image 输入
- **AND** batch MUST 包含 beam label 或现有 canonical target beam label
- **AND** batch MUST 在可用时包含 `scene`、`sample_id` 和 `split`
- **AND** batch MUST NOT 包含 `gps`、`lidar`、`radar`、`mmwave`、`csi`、`channel`、`path` 或 `beam_power` 作为可被模型、loss 或 adaptation 消费的字段

#### Scenario: collate 不要求禁用模态 key
- **WHEN** image-only dataset sample 不包含 GPS、LiDAR、radar、mmWave、CSI、channel、path 或 beam_power key
- **THEN** collate MUST 成功构造 batch
- **AND** dataloader one batch smoke test MUST 能在 `conda run -n kd_mm_beam` 环境中通过

#### Scenario: 原始字段存在不等于被消费
- **WHEN** 原始样本文件或 manifest 记录 path、radio、channel、beam_power、GPS 或 LiDAR 可用性
- **THEN** image-only batch preparation MUST 不把这些字段传给模型、loss、adaptation 或 evaluator
- **AND** run metadata MUST 区分 `available_fields` 与 `consumed_fields`

### Requirement: Image-only probe label 使用边界
image-only legal probe 的 dataset 和 batch preparation MUST 只把 beam label 作为 supervised target 暴露给 source training、target support adaptation 和 target_test evaluation。target adaptation MUST NOT 暴露 target test label、path/radio/channel label、beam_power argmax 或任何禁用 oracle 字段。

#### Scenario: target support 暴露 beam label
- **WHEN** 构建 target support dataloader 或 support feature cache
- **THEN** batch/cache MUST 暴露 support beam label
- **AND** support label source MUST 记录到 sampling metadata

#### Scenario: target test label 只用于 evaluation
- **WHEN** 构建 target test dataloader 或 target_test feature cache
- **THEN** target test beam label MUST 只在 evaluation scope 用于指标计算
- **AND** adaptation、threshold selection、temperature fitting、target prior 初始化和 prototype 构建 MUST NOT 读取 target test beam label

#### Scenario: 禁用 beam_power 离线指标时记录 unavailable
- **WHEN** image-only probe 禁用 `beam_power`
- **THEN** evaluation MUST NOT 为 BPL dB 或 NRP 读取 beam_power 作为输入或 adaptation 信号
- **AND** 如果无法合法计算 BPL dB 或 NRP，metrics MUST 将对应指标标记为 unavailable 并记录原因
