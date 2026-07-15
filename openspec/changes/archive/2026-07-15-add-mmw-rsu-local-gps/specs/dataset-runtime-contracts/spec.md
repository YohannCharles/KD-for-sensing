## ADDED Requirements

### Requirement: GPS 坐标系 runtime metadata
启用 GPS 的训练和评估 runtime MUST 记录实际 `gps_feature_mode`；当模式为 `rsu_local_relative_polar` 时，还 MUST 记录 RSU yaw 来源、输入序列是否逐帧具有有限 yaw，以及 train/validation/test mode 一致性。该 metadata MUST 可用于阻止不同坐标系的 scaler、sample cache 或 checkpoint 评估配置静默混用。

#### Scenario: MMW 局部 GPS metadata 完整
- **WHEN** MMW dataloader 使用 `rsu_local_relative_polar`
- **THEN** runtime metadata MUST 记录 `gps_feature_mode=rsu_local_relative_polar`
- **AND** metadata MUST 记录 `gps_angle_frame=rsu_local` 和 `gps_yaw_source=bs_yaml:sensors.rsu_pose.rotation.yaw`
- **AND** metadata MUST 记录当前 split 的 yaw validation 状态

#### Scenario: 保持世界坐标 GPS metadata
- **WHEN** dataloader 使用既有 `relative_polar`
- **THEN** runtime metadata MUST 记录 `gps_feature_mode=relative_polar`
- **AND** 系统 MUST 不要求读取或记录 RSU yaw 才能构建旧模式

#### Scenario: 坐标系相关 cache 不匹配
- **WHEN** resolved config 请求的 GPS feature mode 与 sample cache 或 scaler metadata 中的 mode 不一致
- **THEN** runtime MUST 拒绝该 artifact 或重新生成 mode 隔离的 artifact
- **AND** runtime MUST 不把世界坐标 GPS tensor 当作 RSU 局部 GPS tensor 使用
