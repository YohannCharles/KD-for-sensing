## ADDED Requirements

### Requirement: DeepSense6G Top8 selector optional modality loading
DeepSense6G Top8 candidate dataset MUST 按配置和 manifest availability 加载 optional modalities。未启用或不可用的 camera AE、image tensor、LiDAR feature 和 radar feature MUST 不阻止 GPS context-only selector 运行；启用某个 optional modality 时，dataset MUST 只读取该模态需要的 path 或 feature，不触发其它模态 IO。

#### Scenario: GPS context-only selector 不读取图像或点云
- **WHEN** 配置运行 `gps_context_only_selector`
- **THEN** dataset MUST 读取 Top8 candidate manifest、candidate fields 和 GPS context fields
- **AND** dataset MUST NOT 读取 image file、camera AE feature、LiDAR feature 或 radar feature
- **AND** 返回样本 MUST 不包含未启用 optional modality 的大张量

#### Scenario: camera AE 可用时按 row index 读取
- **WHEN** 配置启用 camera AE feature 且 manifest 包含有效 `camera_ae_feature_row_index`
- **THEN** dataset MUST 从配置的 AE feature artifact 读取对应 feature row
- **AND** 返回样本 MUST 包含 `camera_ae_feature`
- **AND** dataset MUST 在 metadata 中记录 AE feature artifact path 或 fingerprint

#### Scenario: camera AE 缺失时记录原因
- **WHEN** 配置启用 camera AE feature 但 manifest 中 feature row index 无效或 artifact 缺失
- **THEN** dataset MUST 返回缺失标记
- **AND** runner MUST 跳过 camera AE 相关 ablation 或降级到 GPS context-only selector
- **AND** summary MUST 写入 `skipped_reason`

#### Scenario: image/LiDAR/radar feature 按需读取
- **WHEN** 配置启用 image tensor、LiDAR feature 或 radar feature
- **THEN** dataset MUST 只读取对应模态字段中声明的 path 或 feature
- **AND** 其它未启用模态 MUST 不触发 path 解析、cache 初始化或文件读取

### Requirement: Top8 selector normalization fit boundary
Top8 selector dataset MUST 支持为 candidate features 和 GPS context 保存 normalization metadata。E、N、log_range、speed、candidate logits 等统计量 MUST 只从允许训练的 source/support 样本拟合，target query 样本 MUST 不参与 fit。

#### Scenario: support/source fit scaler
- **WHEN** dataset 构建 normalization artifact
- **THEN** scaler fit MUST 只使用 source training rows、target support rows 或 target support internal train rows
- **AND** metadata MUST 记录 fit split、样本数、字段名和随机种子

#### Scenario: query 不参与 normalization fit
- **WHEN** manifest 中包含 target query rows
- **THEN** target query rows MUST 只使用已经拟合好的 normalization 参数进行 transform
- **AND** target query label 或 query 统计量 MUST NOT 影响 scaler 参数
