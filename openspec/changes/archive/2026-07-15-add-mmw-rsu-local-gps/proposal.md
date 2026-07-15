## Why

Town03 五场景的 RSU 阵列朝向不同，但当前 MMW `relative_polar` GPS 输入始终使用世界坐标角。五场景合并训练时，同一世界角会对应不同 beam 区域，导致 GPS 分支出现可避免的跨场景冲突；需要增加显式、可审计的 RSU 局部坐标模式，并用严格配对实验确认其实际收益。

## What Changes

- 新增向后兼容的 `rsu_local_relative_polar` GPS 特征模式，仅对带 RSU pose yaw 的 MMW YAML 输入开放；原 `relative_polar` 数值与旧配置保持不变。
- 从每个历史时隙的 BS YAML 读取 `sensors.rsu_pose.rotation.yaw`，将 UE-BS 相对向量旋转到 RSU 局部坐标后继续输出 `[dist, sin_theta, cos_theta]`。
- 缺失、非法或不一致的 RSU yaw 必须在数据加载阶段清晰失败；不得静默使用 camera、LiDAR 或场景名推断的朝向。
- MMW all-weather launcher 在启动 GPU 前必须检查 `bs_gps1..5`、逐引用 yaw 和每 domain 静态 yaw provenance。
- 在 runtime/config provenance 中记录 GPS feature mode 与 yaw 来源，使 checkpoint 和评估能够区分世界坐标与 RSU 局部坐标输入。
- 使用相同 Town03 五场景、三天气、H5/P1 split、seed、训练预算、缺失协议和 `last.pth` 规则，配对比较修复前后的 GPS 表征诊断、T2 clean 与缺失模态结果。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `gps-preprocessing`: 在保留 `relative_polar` 默认行为的同时，增加 MMW RSU 局部相对极坐标模式及 yaw 缺失时的失败契约。
- `dataset-runtime-contracts`: 要求运行时记录实际 GPS 坐标系模式和 RSU yaw 来源，并保证 train/validation/test 使用一致的数据契约。

## Impact

- GPS 特征与合同：`src/kd_sensing/data/transform_ops/gps.py`、DeepSense/MMW dataset GPS contract helper 及 focused tests。
- MMW all-weather 本地实验：复用现有 launcher/evaluator，生成配置、日志、checkpoint 和结果只写入 ignored `outputs/`。
- 不新增模型结构、第三方依赖、公共 CLI 或实体主配置；不修改 beam label、split、mask cache 和已有 checkpoint。
