## ADDED Requirements

### Requirement: TinyViT preset 注册可表驱动
TinyViT encoder 的四个公开注册名 MAY 通过单一 preset 表和循环注册实现。实现 MUST 保持注册名、variant、pretrained 标志、pretrained_source、metadata 和构建参数覆盖语义不变。

#### Scenario: 四个 TinyViT 注册名保持可用
- **WHEN** 构建流程调用默认组件导入后查看 `ENCODERS.list()`
- **THEN** 列表 MUST 包含 `tinyvit_5m_scratch_rgb`、`tinyvit_5m_22k_rgb`、`tinyvit_11m_scratch_rgb` 和 `tinyvit_11m_22k_rgb`
- **AND** 每个注册名 MUST 能通过 `ENCODERS.build()` 构建对应 TinyViT image encoder

#### Scenario: preset 注册不新增抽象层
- **WHEN** TinyViT 注册从复制粘贴调用改为表驱动循环
- **THEN** 实现 MUST 不新增通用 registry wrapper、plugin 系统或额外 factory class
- **AND** 未知 TinyViT 名称 MUST 继续使用现有 registry 错误风格
