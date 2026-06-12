## REMOVED Requirements

### Requirement: GPS 滑动窗口 baseline 输入边界
**Reason**: GPS window 非神经几何 baseline 不再属于 Image+GPS JEPA query-pool 主线或必要 paired control，本轮删除对应源码、CLI 和配置。
**Migration**: 使用保留的 Image+GPS supervised/GPS-biased/GPS-query JEPA 配置，或使用 vision-position baseline suite 中的 `gps_only_neural` 作为 GPS-only 对照。

### Requirement: GPS 几何滤波与 beam score 输出
**Reason**: GPS window baseline 已退役，不再维护几何滤波算法族、beam score tensor 或参数化非神经预测输出。
**Migration**: 如需 GPS-only 对照，使用当前 vision-position baseline suite；如需历史几何 baseline，可从 git 历史恢复退役代码进行离线复查。

### Requirement: 全场景 GPS baseline 评估
**Reason**: `kd-sensing-gps-window-baseline`、`src/kd_sensing/baselines/gps_window/` 和 `configs/baselines/gps_window_*.yaml` 退役删除，不再提供当前 all-scenes profile。
**Migration**: 当前评估矩阵以 Image+GPS JEPA query-pool paired baseline/control 和 vision-position baseline suite 为主。

### Requirement: 参数搜索与逐轮调参记录
**Reason**: GPS window sweep/calibration workflow 不再作为当前实验路线维护。
**Migration**: 不迁移；历史调参产物保留在本地 outputs/logs 时仍按 runtime artifact cleanup manifest 管理。

### Requirement: GPS baseline 防泄漏与可审计产物
**Reason**: 该防泄漏契约仅服务已退役 GPS window baseline。
**Migration**: 当前训练/评估防泄漏继续由 dataset split、target-shot splitting、runtime metadata 和配置验证能力覆盖。
