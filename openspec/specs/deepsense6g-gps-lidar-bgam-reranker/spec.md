# deepsense6g-gps-lidar-bgam-reranker Specification

## Purpose
记录 DeepSense6G GPS+LiDAR BGAM reranker workflow 退役后的防回流边界，确保旧配置、入口、模型、loss、manifest enrich、debug mask 和 focused tests 不再被误认为当前能力。

## Requirements
### Requirement: DeepSense6G GPS+LiDAR BGAM reranker 已退役
DeepSense6G GPS+LiDAR BGAM reranker 不再属于当前支持能力。系统 MUST 删除该 workflow 的配置、console scripts、包内 CLI、engine、model、loss、dataset、manifest helper、debug mask/report helper 和 focused tests；系统 MUST NOT 提供兼容 stub、thin alias、virtual config 或迁移后门来继续运行该 workflow。

#### Scenario: 旧 BGAM 配置和入口不存在
- **WHEN** 开发者检查仓库配置、安装入口和包内 CLI
- **THEN** 项目 MUST 不保留 `configs/deepsense6g_gps_lidar_bgam.yaml`
- **AND** 项目 MUST 不声明 DeepSense6G GPS+LiDAR BGAM prepare/run/evaluate 相关 `kd-sensing-*` 命令
- **AND** 包内 MUST 不保留 `kd_sensing.cli.*gps_lidar_bgam*` 入口模块

#### Scenario: BGAM 运行实现不存在
- **WHEN** 开发者检查 source tree 和 import surface
- **THEN** 项目 MUST 不保留 DeepSense6G BGAM engine、dataset、manifest、model 或 loss 模块
- **AND** 导入旧 `kd_sensing.*gps_lidar_bgam*` 模块 MUST 失败
- **AND** 项目 MUST 不保留旧 BGAM geometry/model/dataset/runner focused tests

#### Scenario: 旧配置快速失败
- **WHEN** 用户传入旧 BGAM config path、experiment name、model type 或 override key/value
- **THEN** migration guard MUST 早失败
- **AND** 错误信息 MUST 说明 BGAM 已退役且不再支持兼容入口

### Requirement: 通用几何与指标能力不因 BGAM 退役而删除
BGAM 退役只删除该 workflow 专属支持面。系统 MUST 保留仍被当前能力使用的通用 GPS/RSU geometry、64-beam circular metric、Top-K metric、MMW GPS v2、CSI candidate ranking 和 JEPA 诊断能力；这些能力 MUST NOT 通过旧 BGAM module path、config path 或 console script 暴露。

#### Scenario: 当前能力不依赖 BGAM 模块名
- **WHEN** 当前 GPS v2、CSI、JEPA 或通用评估路径需要角度、Top-K 或 circular metric helper
- **THEN** 实现 MUST 使用当前通用 helper 或当前 owner module
- **AND** 实现 MUST NOT 重新引入 `gps_lidar_bgam`、`lidar_bgam` 或 BGAM 专属 module path
