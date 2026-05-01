## ADDED Requirements

### Requirement: Image cache 感知加载
Scenario 9 dataset 在启用 image modality 时 MUST 支持从 image motion mask cache 懒加载预处理结果。cache 不存在且未启用写入时，系统 MUST 保持旧的在线计算路径或按配置抛出清晰错误。未启用 image modality 时，dataset MUST 不检查、不读取、不写入 image cache。

#### Scenario: image-only 使用 motion cache
- **WHEN** 用户运行 image-only 配置且启用 `image_motion_use_cache`
- **THEN** dataset MUST 对当前样本所需相邻帧 pair 尝试读取 motion mask cache
- **AND** 返回样本 MUST 继续只包含 image 和 label 字段

#### Scenario: fusion 只为启用 image 的配置使用 cache
- **WHEN** 用户运行 fusion 配置且 `modalities` 不包含 image
- **THEN** dataset MUST 不访问 image motion cache 配置
- **AND** 缺失 image cache 或 jpg 文件不得阻止该非 image fusion 任务运行

### Requirement: Beam label 轻量缓存
Scenario 9 dataset MUST 支持在当前 split 内缓存 beam 文本解析结果。该缓存 MUST 是轻量整数映射，不得把 image、radar、GPS 或 LiDAR 大数组放入同一初始化缓存中。

#### Scenario: Dataset 初始化建立 beam label cache
- **WHEN** 配置启用 eager beam label cache
- **THEN** dataset MAY 扫描当前 split 唯一 input/future beam 路径并保存整数 label 映射
- **AND** 初始化缓存 MUST 只保存 path 和 int label，不保存大模态数组

#### Scenario: Dataset 按需建立 beam label cache
- **WHEN** 配置使用 lazy beam label cache
- **THEN** dataset MUST 在第一次遇到 beam path 时解析并缓存 label
- **AND** 后续遇到同一路径 MUST 复用缓存 label
