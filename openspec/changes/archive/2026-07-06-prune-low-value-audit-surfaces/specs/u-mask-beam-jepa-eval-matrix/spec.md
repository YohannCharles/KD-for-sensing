## ADDED Requirements

### Requirement: Eval matrix export writer 归属评估矩阵 owner
U-mask Beam JEPA eval matrix 的 CSV、JSON 和 Markdown 写出逻辑 MUST 归属于 eval matrix owner 或其窄 helper。项目 MUST 不保留只为该矩阵服务的 `kd_sensing.eval.export` 聚合模块作为 current API。

#### Scenario: CLI 使用 eval matrix owner writer
- **WHEN** `kd-sensing-eval-u-mask-beam-jepa-matrix` 写出 CSV、JSON 或 Markdown 结果
- **THEN** CLI glue MUST 调用 U-mask Beam JEPA eval matrix owner 或其窄 helper 的 writer
- **AND** CLI MUST 不依赖 `kd_sensing.eval.export` 作为通用 writer facade

#### Scenario: trainer runtime helper 不导入小聚合模块
- **WHEN** training runtime helper 需要导出缺失矩阵或评估矩阵结果
- **THEN** 它 MUST 从 eval matrix owner 导入 evaluation 和 writer helper，或把写出逻辑局部化
- **AND** 它 MUST 不通过 `kd_sensing.eval.export` 维持旧 helper path

#### Scenario: 导出格式行为保持
- **WHEN** 实现删除 `kd_sensing.eval.export`
- **THEN** U-mask Beam JEPA eval matrix 的 CSV、JSON 和 Markdown 输出字段、排序和无副作用测试行为 MUST 保持一致
- **AND** focused tests MUST 覆盖至少一个 CSV/JSON/Markdown 写出路径或等价 formatter 行为
