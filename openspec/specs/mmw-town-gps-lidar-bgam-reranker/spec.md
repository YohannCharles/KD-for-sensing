# mmw-town-gps-lidar-bgam-reranker Specification

## Purpose
记录 MMW Town GPS+LiDAR BGAM reranker workflow 退役后的防回流边界，避免旧 pseudo-history、LiDAR BGAM gate、Top8 candidate rerank 和评估产物继续作为当前 workflow 入口出现。

## Requirements
### Requirement: MMW Town GPS+LiDAR BGAM reranker 已退役
MMW Town GPS+LiDAR BGAM reranker 不再属于当前支持能力。系统 MUST 删除该 workflow 的配置、console scripts、包内 CLI、engine、manifest helper、model/loss 依赖和 focused tests；系统 MUST NOT 提供兼容 stub、thin alias、virtual config 或旧输出目录作为当前入口。

#### Scenario: 旧 MMW BGAM 配置和入口不存在
- **WHEN** 开发者检查配置、安装入口和包内 CLI
- **THEN** 项目 MUST 不保留 `configs/mmw_town_gps_lidar_bgam.yaml`
- **AND** 项目 MUST 不声明 MMW Town GPS+LiDAR BGAM prepare/run/evaluate 相关 `kd-sensing-*` 命令
- **AND** 包内 MUST 不保留 `kd_sensing.cli.*mmw_town*gps_lidar_bgam*` 入口模块

#### Scenario: 旧 pseudo-history BGAM 运行实现不存在
- **WHEN** 开发者检查 source tree 和 import surface
- **THEN** 项目 MUST 不保留 MMW BGAM engine 或 manifest helper
- **AND** 项目 MUST 不通过 `gps_lidar_bgam`、`lidar_bgam` 或 `oracle_history_bgam_upper_bound` 恢复旧 workflow
- **AND** 导入旧 MMW BGAM module path MUST 失败

#### Scenario: MMW 当前路线不借 BGAM 复活
- **WHEN** 用户运行 MMW GPS-only v2、group-safe split、label-space calibration、CSI hardening 或当前诊断
- **THEN** 这些路线 MUST 不生成 BGAM candidate manifest、BGAM mask/debug report 或 BGAM rerank checkpoint
- **AND** 旧 MMW BGAM 输出只能作为历史本地产物或清理候选，不得作为当前文档命令来源
