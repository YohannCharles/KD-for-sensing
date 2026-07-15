## MODIFIED Requirements

### Requirement: GPS-Rel-Polar 特征模式
系统 MUST 支持 `relative_polar` 作为默认 GPS 特征模式，并 MUST 支持 MMW YAML 专用的 `rsu_local_relative_polar` opt-in 模式。两个模式 MUST 对每个历史时隙输出 `[dist, sin_theta, cos_theta]` 三维特征；`relative_polar` MUST 保持既有世界坐标角语义，`rsu_local_relative_polar` MUST 使用 UE-BS 相对向量减去同一时隙 RSU pose yaw 后的局部角。系统 MUST 拒绝其它未声明 GPS 特征模式。

#### Scenario: 构造默认 GPS-Rel-Polar 特征
- **WHEN** `gps_feature_mode` 为 `relative_polar` 或 GPS 配置未显式设置特征模式
- **THEN** 系统 MUST 基于 UE-BS 世界坐标相对向量输出 `[dist, sin_theta, cos_theta]` 三维特征
- **AND** 系统 MUST 使用 `sin_theta` 和 `cos_theta` 表示角度，避免直接输出有跳变的角度值
- **AND** 输出 MUST 与本变更前的 `relative_polar` 数值兼容

#### Scenario: 构造 MMW RSU 局部相对极坐标特征
- **WHEN** `gps_feature_mode` 为 `rsu_local_relative_polar` 且 UE/BS 输入为包含 RSU pose yaw 的 MMW YAML 序列
- **THEN** 系统 MUST 从每个 `bs_gpsN` YAML 读取 `sensors.rsu_pose.rotation.yaw`
- **AND** 系统 MUST 计算 `theta_local = atan2(ue_y-bs_y, ue_x-bs_x) - yaw_rsu`
- **AND** 输出 MUST 为 `[dist, sin(theta_local), cos(theta_local)]` float32 数组且 shape 为 `[seq_len, 3]`

#### Scenario: 局部模式缺少权威 RSU yaw
- **WHEN** `rsu_local_relative_polar` 输入不是 MMW YAML，或任一 BS YAML 缺少有限的 `sensors.rsu_pose.rotation.yaw`
- **THEN** 系统 MUST 在数据加载阶段失败并报告具体资源路径
- **AND** 系统 MUST 不使用 camera、LiDAR、场景名常量或默认零角替代

#### Scenario: 静态 RSU 窗口内 yaw 不一致
- **WHEN** `rsu_local_relative_polar` 的同一历史窗口内 RSU yaw 超出固定数值容差
- **THEN** 系统 MUST 在数据加载阶段失败并报告相关 BS YAML 路径与 yaw 值
- **AND** 系统 MUST 不对不一致 yaw 求平均或只取最后一帧

#### Scenario: 拒绝非保留 GPS 特征模式
- **WHEN** 用户配置 `gps_feature_mode` 为 `raw`、`utm`、`relative`、`motion`、`motion_smooth` 或其它未声明值
- **THEN** 系统 MUST 拒绝启用该 GPS 配置
- **AND** 错误信息 MUST 列出 `relative_polar` 与 `rsu_local_relative_polar` 的适用范围

## ADDED Requirements

### Requirement: MMW RSU yaw 与 scaler provenance
MMW RSU 局部 GPS 模式 MUST 记录 feature mode、yaw source 和训练集 scaler provenance。训练、验证和评估 MUST 使用相同 feature mode；scaler MUST 只由对应 mode 的训练 split 拟合并复用于 validation/test。

#### Scenario: 保存局部 GPS scaler
- **WHEN** 训练使用 `rsu_local_relative_polar` 且启用 GPS normalization
- **THEN** 保存的 scaler metadata MUST 标记 feature mode 为 `rsu_local_relative_polar`
- **AND** yaw source MUST 记录为 `bs_yaml:sensors.rsu_pose.rotation.yaw`
- **AND** scaler 数值工件 MUST 不包含 RSU yaw 或 validation/test 统计量

#### Scenario: 拒绝 world/local scaler 混用
- **WHEN** evaluation 配置使用 `rsu_local_relative_polar` 但加载的 scaler provenance 标记为 `relative_polar`
- **THEN** 系统 MUST fail closed 或要求重新提供匹配 mode 的训练 scaler
- **AND** 系统 MUST 不静默转换或复用不匹配 scaler

### Requirement: MMW pooled 局部 GPS preflight 与配对验证
MMW pooled all-weather 实验使用 `rsu_local_relative_polar` 时 MUST 在启动训练前验证全部 domain 的 BS GPS 列、引用 YAML 和静态 RSU yaw，并 MUST 使用与 world 坐标对照相同的数据、模型和训练评估协议。

#### Scenario: 15-domain 局部 GPS preflight
- **WHEN** all-weather launcher 请求 `rsu_local_relative_polar`
- **THEN** preflight MUST 要求每个 split 具有 `bs_gps1..bs_gps5`
- **AND** preflight MUST 逐引用验证 `sensors.rsu_pose.rotation.yaw` 为有限值且窗口内一致
- **AND** preflight 输出 MUST 记录每个 domain 的 yaw、feature mode、angle frame 和 yaw source
- **AND** 任一 domain 失败时 launcher MUST 不启动 GPU 训练

#### Scenario: world/local matched T2 对照
- **WHEN** 系统比较 `relative_polar` 与 `rsu_local_relative_polar` T2
- **THEN** 两个 resolved config 除 GPS mode、GPS input profile、output identity 和对应 provenance 外 MUST 保持一致
- **AND** 两个 run MUST 使用相同 15-domain inventory、H5/P1 split、seed、训练预算、domain-balanced sampler、missing augmentation 和 fixed-epoch `last.pth` policy
- **AND** missing evaluation MUST 复用相同 sample 与 mask cache identity
