## ADDED Requirements

### Requirement: DeepSense6G GPS Top8 candidate selector 已退役
DeepSense6G GPS Top8 candidate selector 训练/plot/compare workflow 不再属于当前支持能力。系统 MUST 不再提供 selector/attention selector、runner、plotter、comparison report 或默认配置。BGAM 依赖的 TopK candidate manifest/dataset/loss 支撑代码 MAY 保留。

#### Scenario: Top8 selector 入口不存在
- **WHEN** 开发者检查 console scripts、配置和包内模块
- **THEN** 项目 MUST 不声明 DeepSense6G Top8 selector 相关 `kd-sensing-*` 命令
- **AND** 项目 MUST 不保留 `configs/deepsense6g_top8_selector.yaml`
- **AND** 项目 MUST 不保留 Top8 selector 专属 model、engine 或 tests

## REMOVED Requirements

### Requirement: DeepSense6G GPS Top8 candidate selector workflow
**Reason**: GPS Top8 内重排路线已被实验判定不可行，已退役。
**Migration**: 无兼容迁移。

### Requirement: TopKCandidateSelector model
**Reason**: selector model 已退役。
**Migration**: 无兼容迁移。

### Requirement: CandidateAttentionSelector model
**Reason**: attention selector 已退役。
**Migration**: 无兼容迁移。

### Requirement: Training protocol and ablation matrix
**Reason**: Top8 selector training protocol 已退役。
**Migration**: 无兼容迁移。

### Requirement: Top8 selector evaluation artifacts
**Reason**: Top8 selector 输出产物已退役。
**Migration**: 无兼容迁移。

### Requirement: Top8 selector visualization and comparison report
**Reason**: Top8 selector plot/compare workflow 已退役。
**Migration**: 无兼容迁移。

### Requirement: Top8 selector validation and documentation
**Reason**: Top8 selector 不再作为当前可验证能力。
**Migration**: 使用保留 workflow 的 validation。
