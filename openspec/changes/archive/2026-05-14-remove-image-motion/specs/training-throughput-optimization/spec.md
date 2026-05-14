## ADDED Requirements

### Requirement: Image 路径不产生 motion cache
训练吞吐优化 MUST 不再依赖 image motion mask cache。包含 image modality 的 profile、训练和评估运行 MUST 直接衡量 RGB/ImageNet image 加载与模型 step，而不是预热、读取或写入 image motion cache。

#### Scenario: image profile 不报告 motion cache
- **WHEN** 用户运行包含 image modality 的训练 I/O profile
- **THEN** profile 输出 MUST 记录 image modality 使用 RGB/ImageNet 输入
- **AND** profile 输出 MUST 不包含 image motion cache hit、miss、write 或 cache 目录指标

#### Scenario: 旧 image motion 预处理入口不可用
- **WHEN** 用户运行 image motion mask cache 预处理入口或加载 `configs/preprocess/image_motion_cache.yaml`
- **THEN** 系统 MUST 拒绝该操作
- **AND** 错误信息 MUST 说明 image motion cache 已删除且不会生成替代 cache

## REMOVED Requirements

### Requirement: Image motion mask cache
**Reason**: image motion mask 方法适用面较窄，且与 RGB/ImageNet image 主路径并存会持续增加配置、缓存和诊断复杂度。
**Migration**: 删除旧 cache 和配置依赖，直接使用 RGB/ImageNet image 输入重新训练；历史 `outputs/` 不迁移。

#### Scenario: 从 cache 读取 motion mask
- **WHEN** image modality 启用且旧配置启用 `image_motion_use_cache`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不尝试读取 motion mask cache 文件

#### Scenario: cache miss 在线生成并写入
- **WHEN** image modality 启用且旧配置启用 `image_motion_write_cache`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不在线生成或写入 motion mask cache

#### Scenario: 参数变化不误用旧 cache
- **WHEN** 旧配置包含 image motion cache 参数
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不创建参数 hash cache 目录

#### Scenario: 预处理入口预热 image cache
- **WHEN** 用户请求预热 image motion mask cache
- **THEN** 系统 MUST 拒绝该预处理类型
- **AND** 系统 MUST 不写出 image motion cache metadata
