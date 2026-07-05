## ADDED Requirements

### Requirement: Shared runtime profile routing focused test
项目健康护栏 MUST 覆盖 shared runtime 的单模态 input profile routing。新增或修改 `prepare_task_inputs`、`prepare_fusion_inputs` 或单模态 input preparation helper 时，focused tests MUST 验证 profile key 与 modality 一致，并且测试 MUST 不读取真实 `dataset/`、不启动训练、不写入 checkpoint。

#### Scenario: 单模态 profile 路由回归被测试发现
- **WHEN** 开发者运行 runtime profile routing focused test
- **THEN** 测试 MUST 覆盖 radar、gps 和 lidar 的 profile 透传
- **AND** 如果任一单模态任务读取其它 modality 的 profile，测试 MUST 失败并指出任务名和错误 profile key

#### Scenario: runtime 改动后的最小验证
- **WHEN** 变更触碰 shared runtime input preparation
- **THEN** tasks 或最终说明 MUST 至少列出对应 runtime focused test
- **AND** Python 验证命令 MUST 使用 `conda run -n kd_mm_beam`
