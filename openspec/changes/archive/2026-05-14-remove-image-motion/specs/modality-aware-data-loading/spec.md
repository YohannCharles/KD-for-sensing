## ADDED Requirements

### Requirement: RGB image 感知加载
Scenario 9 dataset 在启用 image modality 时 MUST 直接加载 RGB/ImageNet image 输入。系统 MUST 不再支持 motion mask 在线生成、motion mask cache 懒加载或 `motion_mask` profile。

#### Scenario: image-only 使用 RGB 输入
- **WHEN** 用户运行 image-only 配置
- **THEN** dataset MUST 读取当前样本所需的 RGB image 帧
- **AND** 返回样本 MUST 包含 `image` 和 label 字段
- **AND** 返回的 `image` MUST 可被 RGB/ImageNet image encoder 消费
- **AND** dataset MUST 不调用 motion mask 生成或 image motion cache 路径解析逻辑

#### Scenario: motion profile 被拒绝
- **WHEN** 用户配置 `image_profile: motion_mask`
- **THEN** dataset 或配置解析 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明 image motion 路径已删除且需要使用 RGB/ImageNet image 输入

## MODIFIED Requirements

### Requirement: Scenario 9 按模态选择加载样本
Scenario 9 dataset MUST 根据训练或评估配置中的启用模态加载样本字段。未启用模态的文件 MUST 不被读取，未启用模态的输入字段 MUST 不出现在样本字典中，且未启用模态的路径列或文件缺失不得阻止当前任务运行。dataset MUST 始终加载 beam 历史标签和 future beam 目标标签。

#### Scenario: GPS-only 不读取 image 或 radar 文件
- **WHEN** 用户运行 `experiment.task: gps` 的训练或评估配置
- **THEN** dataset MUST 只读取 GPS、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image 或 radar map 加载逻辑
- **AND** 返回样本 MUST 不包含 `image`、`radar_ra` 或 `radar_da`

#### Scenario: LiDAR-only 不读取 image 或 radar 文件
- **WHEN** 用户运行 `experiment.task: lidar` 的训练或评估配置
- **THEN** dataset MUST 只读取 LiDAR、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、radar map 或 GPS 加载逻辑
- **AND** 返回样本 MUST 不包含 `image`、`radar_ra`、`radar_da` 或 `gps`

#### Scenario: mmWave-only 不读取其它输入模态文件
- **WHEN** 用户运行 `experiment.task: mmwave` 的训练或评估配置
- **THEN** dataset MUST 只读取 mmWave、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、radar map、GPS 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 包含 `mmwave`
- **AND** 返回样本 MUST 不包含 `image`、`radar_ra`、`radar_da`、`gps` 或 `lidar`

#### Scenario: radar-only 只读取 radar 输入
- **WHEN** 用户运行 `experiment.task: radar` 的训练或评估配置
- **THEN** dataset MUST 只读取 radar、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、GPS、LiDAR 或 mmWave 加载逻辑
- **AND** 返回样本 MUST 包含 `radar_ra` 和 `radar_da`

#### Scenario: image-only 只读取 image 输入
- **WHEN** 用户运行 `experiment.task: image` 的训练或评估配置
- **THEN** dataset MUST 只读取 RGB image、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 radar、GPS、LiDAR、mmWave 或 image motion cache 逻辑
- **AND** 返回样本 MUST 包含 `image`

#### Scenario: fusion 按 modalities 读取输入
- **WHEN** 用户运行 `experiment.task: fusion` 且配置 `modalities: ["radar", "mmwave"]`
- **THEN** dataset MUST 只读取 radar、mmWave、`input_beam` 和 `target_beam` 所需文件
- **AND** dataset MUST 不调用 image、GPS 或 LiDAR 加载逻辑
- **AND** 返回样本 MUST 只包含启用模态对应输入字段和标签字段

### Requirement: 自动 cache policy 下的模态感知 cache 访问
Scenario 9 dataset MUST 在自动 cache policy 下保持按模态访问数据。启用 LiDAR 时允许使用 LiDAR BEV cache；未启用 LiDAR 时 MUST 完全跳过 LiDAR cache 访问。启用 image 时 MUST 不访问 image motion cache。

#### Scenario: image-only 不使用 image motion cache
- **WHEN** 用户运行 image-only 配置且 `data.cache.policy: auto`
- **THEN** dataset MUST 使用 RGB/ImageNet image 输入
- **AND** dataset MUST 不读取、不创建、不写入 image motion cache
- **AND** 返回样本字段、shape 和 dtype MUST 与 RGB/ImageNet image 契约一致

#### Scenario: LiDAR fusion 自动使用 LiDAR cache
- **WHEN** 用户运行包含 LiDAR 的 fusion 配置且 `data.cache.policy: auto`
- **THEN** dataset MUST 对 LiDAR BEV 启用 cache 读取
- **AND** cache miss 时 dataset MUST 生成并写入缺失的 LiDAR BEV cache
- **AND** 返回样本字段、shape 和 dtype MUST 与未启用 cache 时一致

#### Scenario: 非相关模态不触发 cache 初始化
- **WHEN** 用户运行不包含 LiDAR 的单模态或 fusion 配置
- **THEN** dataset 初始化 MUST 不创建 LiDAR cache 目录
- **AND** dataset 取样 MUST 不调用 LiDAR cache path 解析逻辑
- **AND** dataset MUST 不调用任何 image motion cache path 解析逻辑

## REMOVED Requirements

### Requirement: Image cache 感知加载
**Reason**: image motion mask cache 已被删除，image modality 不再有独立的 motion cache artifact。
**Migration**: 使用 RGB/ImageNet image 输入；重新运行实验生成新的 checkpoint 和评估结果。

#### Scenario: image-only 使用 motion cache
- **WHEN** 用户运行旧 image-only 配置且启用 `image_motion_use_cache`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出 `image_motion_use_cache` 已删除

#### Scenario: fusion 只为启用 image 的配置使用 cache
- **WHEN** 用户运行旧 fusion 配置且包含 image motion cache 字段
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出 image motion cache 不再受支持
