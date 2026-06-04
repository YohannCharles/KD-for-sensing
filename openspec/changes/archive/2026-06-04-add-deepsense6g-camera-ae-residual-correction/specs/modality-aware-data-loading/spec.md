## ADDED Requirements

### Requirement: Camera residual manifest data loading
系统 MUST 支持从 camera residual manifest 构建 Dataset/DataLoader。该数据加载路径 MUST 按当前 stage 和 ablation 只读取需要的 image 或 AE feature，并在 image 缺失时保持 GPS-only baseline 可运行。

#### Scenario: gps_prior_only 不读取 image
- **WHEN** ablation 为 `gps_prior_only`
- **THEN** Dataset MUST NOT 读取 image 文件
- **AND** Dataset MUST NOT 要求 `ae_feature_path` 存在
- **AND** 样本 MUST 仍包含 GPS prior、GPS pred、GPS context、target label 和 split role

#### Scenario: AE training 跳过 missing image
- **WHEN** AE training Dataset 读取 manifest
- **THEN** Dataset MUST 只使用 `image_exists=true` 的样本
- **AND** 没有任何可用 image 时 MUST 抛出清晰错误

#### Scenario: residual training 使用 AE feature
- **WHEN** ablation 需要 `camera_ae_feature`
- **THEN** Dataset MUST 根据 `ae_feature_path` 和 `ae_feature_row_index` 读取 feature
- **AND** feature 不可用的样本 MUST 按配置跳过或降级为 GPS context only
- **AND** 降级或跳过原因 MUST 写入 run metadata 或 summary

#### Scenario: query label 不进入训练 batch
- **WHEN** Dataset 为 train/support loader 构建 batch
- **THEN** target query 样本 MUST 不进入训练 batch
- **AND** 如果 evaluation loader 包含 query label，loss 计算 MUST 使用 evaluation-only 路径，不得反向传播或用于 early stopping
