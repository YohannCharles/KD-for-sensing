## ADDED Requirements

### Requirement: 单模态 runtime 按同名 modality 路由 input profile
训练、验证和评估共享 runtime MUST 在单模态任务中使用与任务同名的 `model_cfg.input_profiles` 条目准备输入。`radar` MUST 使用 `input_profiles.radar`，`gps` MUST 使用 `input_profiles.gps`，`lidar` MUST 使用 `input_profiles.lidar`，`mmwave` MUST 使用 `input_profiles.mmwave`，`csi` MUST 使用 `input_profiles.csi`。缺省 profile 仍由对应 modality helper 或 modality contract 解析，不得通过读取其它 modality 的 profile 来补偿。

#### Scenario: radar 单模态使用 radar profile
- **WHEN** runtime 准备 `task: radar` 的单模态 batch，且 `model_cfg.input_profiles` 同时包含 `radar`、`gps` 和 `lidar`
- **THEN** `prepare_radar_inputs` MUST 接收 `input_profiles.radar`
- **AND** runtime MUST NOT 读取 `input_profiles.gps` 或 `input_profiles.lidar` 作为 radar profile

#### Scenario: gps 单模态使用 gps profile
- **WHEN** runtime 准备 `task: gps` 的单模态 batch，且 `model_cfg.input_profiles` 同时包含 `gps` 和 `lidar`
- **THEN** `prepare_gps_inputs` MUST 接收 `input_profiles.gps`
- **AND** runtime MUST NOT 读取 `input_profiles.lidar` 作为 gps profile

#### Scenario: 未声明 profile 时保持默认解析
- **WHEN** 单模态 runtime 准备 batch 且 `model_cfg.input_profiles` 缺少对应 modality
- **THEN** runtime MUST 将缺省 profile 交给对应 modality helper 处理
- **AND** 系统 MUST 不因其它 modality profile 存在而改变该任务的默认 profile
