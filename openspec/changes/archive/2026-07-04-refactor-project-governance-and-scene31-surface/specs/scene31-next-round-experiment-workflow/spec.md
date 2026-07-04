## ADDED Requirements

### Requirement: Scene31 manifest-backed workflow 必须保持生成与运行分离
Scene31 next-round、BC、beamsoft weak、funnel 和 magic overnight workflow MUST 使用 manifest、generator、template/base config 与 local/manual runner 分离表达。训练命令 MUST 继续通过 `kd-sensing-train --config <generated-yaml>` 执行。

#### Scenario: 生成字段保持一致
- **WHEN** 用户通过 Scene31 generator 生成 YAML
- **THEN** run name、seed、epoch、sampler、loss、missing-pattern evaluation 和 output root MUST 与 manifest 行一致
- **AND** generator sanity tests MUST 覆盖这些字段

#### Scenario: runner 复用业务实现
- **WHEN** Scene31 runner 执行 train 或 fresh eval
- **THEN** runner MUST 调用 `conda run -n kd_mm_beam kd-sensing-train` 或现有 apples-to-apples helper
- **AND** runner MUST 不复制 DataLoader、模型加载、指标计算或 checkpoint selection 的业务逻辑
