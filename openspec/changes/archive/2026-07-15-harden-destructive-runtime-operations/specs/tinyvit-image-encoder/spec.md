## ADDED Requirements

### Requirement: TinyViT 远程权重必须验证来源与完整性
TinyViT 预训练权重 MUST 优先使用本地 checkpoint。网络下载默认 MUST 关闭；显式启用下载时 MUST 同时使用固定 HTTPS URL 和预期 SHA256，并在反序列化前完成校验。

#### Scenario: 未校验远程下载被拒绝
- **WHEN** 用户允许下载但 variant 没有固定 HTTPS URL 或预期 SHA256
- **THEN** encoder construction MUST 拒绝下载
- **AND** 错误 MUST 提示提供本地 checkpoint 或完整 hash

#### Scenario: 下载摘要不匹配
- **WHEN** 下载文件的 SHA256 与预期值不同
- **THEN** runtime MUST 拒绝加载并删除不可信临时文件
- **AND** MUST NOT 把该文件发布到共享 cache

#### Scenario: 本地 checkpoint 安全加载
- **WHEN** 用户提供本地 TinyViT checkpoint
- **THEN** runtime MUST 默认使用安全 tensor/state-dict 反序列化模式
- **AND** metadata MUST 记录来源、digest 和加载模式
