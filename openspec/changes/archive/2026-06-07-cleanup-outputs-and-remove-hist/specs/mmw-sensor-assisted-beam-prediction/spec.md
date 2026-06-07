## ADDED Requirements

### Requirement: MMW sensor-assisted 不依赖 Hist
当前 MMW 主线 MUST 不依赖 HiST-Beam/Hist 模型、Hist LOSO executor 或 `configs/hist_beam/`。若保留 MMW sensor-assisted 或 GPS adapter workflow，必须通过非 Hist CLI、配置和模型注册名定义输入、输出和评估。

#### Scenario: MMW 当前主线不构建 Hist 模型
- **WHEN** 用户运行当前 MMW Town GPS v2、GPS adapter、candidate 或其它保留 MMW workflow
- **THEN** 系统 MUST 不构建 `hist_beam_fusion`
- **AND** 输出 metadata MUST 不声明 HiST-Beam variant

#### Scenario: 旧 sensor-assisted Hist 配置不可运行
- **WHEN** 用户引用 `configs/hist_beam/mmw_sensor_assisted_quick_validation.yaml` 或等价 Hist sensor-assisted 配置
- **THEN** 系统 MUST 报告该 Hist 配置已退役或不存在
- **AND** 系统 MUST 不生成 Hist LOSO plan

## REMOVED Requirements

### Requirement: 快速验证实验矩阵
**Reason**: 该矩阵绑定 sensor-assisted HiST variants，已退役。
**Migration**: 当前 MMW workflow 由非 Hist specs 定义自己的矩阵。

### Requirement: History-anchored profile 边界
**Reason**: History-anchored Hist profile 已退役。
**Migration**: 非 Hist history 能力必须重新定义。

### Requirement: History-anchored summary eligibility
**Reason**: History-anchored Hist summary 已退役。
**Migration**: 当前 summary eligibility 由保留 workflow 定义。

### Requirement: History beam 与 sensitive 字段审计
**Reason**: Hist history beam profile 已退役。
**Migration**: 当前敏感字段审计由保留 workflow 定义。
