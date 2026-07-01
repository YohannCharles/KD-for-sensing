## ADDED Requirements

### Requirement: LOSO supporting 语义不绑定 engine dataloader facade
LOSO workflow 的 supporting 契约 MUST 保留 fold planning、target adapt/test split、few-shot sampling 和防泄漏边界，但 MUST 不要求无当前调用方的 `kd_sensing.engine.loso_data` dataloader facade 长期存在。当前或未来可运行 workflow MUST 显式声明自己的 runner、配置和输出契约。

#### Scenario: 删除无调用 engine LOSO helper
- **WHEN** `kd_sensing.engine.loso_data` 的 public builder 无仓库内调用，且 README/docs/OpenSpec current specs、pyproject、tests 和 registry 均不依赖该模块
- **THEN** 本 change MAY 删除或退役该模块
- **AND** `kd_sensing.data.loso` 中的数据集无关 fold/few-shot 规划语义 MUST 保持可用

#### Scenario: 外部兼容风险需要 stub
- **WHEN** 删除前发现 current 文档或用户确认仍有外部脚本 import `kd_sensing.engine.loso_data`
- **THEN** 本 change MUST 保留薄 deprecation stub 或记录后续退役 change
- **AND** stub MUST 不新增训练 runner、adapter loop 或默认 LOSO 执行矩阵

#### Scenario: 未来 LOSO runner 显式建模
- **WHEN** 未来 workflow 需要可运行 LOSO training/adaptation runner
- **THEN** 该 workflow MUST 通过新的 OpenSpec change 声明 CLI、配置矩阵、输出目录和防泄漏验证
- **AND** 系统 MUST 不把已删除或退役的 engine helper 当作隐式默认入口
