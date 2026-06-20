## MODIFIED Requirements

### Requirement: DeepSense6G GPS Top8 candidate selector 已退役
DeepSense6G GPS Top8 candidate selector 训练/plot/compare workflow 不属于当前支持能力。系统 MUST 不再提供 selector/attention selector、runner、plotter、comparison report、默认配置、candidate manifest builder、candidate dataset 或 selector loss 作为当前支撑代码。BGAM 退役后，TopK candidate helper 不再以 BGAM 支撑语义保留。

#### Scenario: Top8 selector 和 BGAM candidate 支撑入口不存在
- **WHEN** 开发者检查 console scripts、配置和包内模块
- **THEN** 项目 MUST 不声明 DeepSense6G Top8 selector 或 BGAM candidate manifest 相关 `kd-sensing-*` 命令
- **AND** 项目 MUST 不保留 `configs/deepsense6g_top8_selector.yaml` 或 BGAM candidate 专属配置
- **AND** 项目 MUST 不保留 Top8 selector 或 BGAM candidate 专属 model、engine、dataset、loss 或 tests

## REMOVED Requirements

### Requirement: Top8 candidate manifest 支撑语义
**Reason**: 该支撑语义主要服务 BGAM；BGAM 退役后不再保留 DeepSense6G Top8 candidate manifest builder。
**Migration**: 无兼容迁移；GPS v2 或其它保留 workflow 不应要求该 manifest。

### Requirement: TopK candidate dataset 支撑语义
**Reason**: TopK candidate dataset helper 在当前支持面中只服务退役 BGAM/selector 路线。
**Migration**: 删除专属 dataset helper；如未来需要候选 dataset，应通过新的 current capability 定义。

### Requirement: TopK candidate selector loss 支撑语义
**Reason**: TopK candidate selector loss 只服务退役 selector/BGAM candidate rerank 路线。
**Migration**: 删除专属 loss；保留通用 Top-K 指标和 circular metrics。
