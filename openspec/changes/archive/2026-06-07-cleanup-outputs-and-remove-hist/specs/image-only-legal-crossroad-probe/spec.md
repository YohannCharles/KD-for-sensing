## ADDED Requirements

### Requirement: Image-only Hist probe 已退役
Image-only legal crossroad probe 中依赖 `configs/hist_beam/`、HiST variants、V8/V9 Hist heads 或 `kd-sensing-hist-beam-loso` 的路径 MUST 从当前支持面退役。

#### Scenario: Image-only Hist probe 配置不可运行
- **WHEN** 用户引用 `configs/hist_beam/image_only_legal_crossroad_probe.yaml`
- **THEN** 系统 MUST 报告配置已退役或不存在
- **AND** 系统 MUST 不构建 image-only HiST probe model

## REMOVED Requirements

### Requirement: Image-only 合法 crossroad probe 配置
**Reason**: 该配置位于已退役 Hist workflow 下。
**Migration**: 当前 image workflow 如需 legal probe，必须用非 Hist 配置重新定义。

### Requirement: Image-only probe 运行矩阵
**Reason**: Image-only Hist run matrix 已退役。
**Migration**: 无兼容迁移。

### Requirement: Image feature cache
**Reason**: Image-only Hist probe cache 契约已退役。
**Migration**: 当前 image feature cache 由保留 workflow 自行定义。

### Requirement: Image-only probe 诊断和汇总产物
**Reason**: Image-only Hist diagnostics 已退役。
**Migration**: 无兼容迁移。

### Requirement: Image-only probe 成功标准
**Reason**: Image-only Hist probe acceptance 已退役。
**Migration**: 无兼容迁移。
