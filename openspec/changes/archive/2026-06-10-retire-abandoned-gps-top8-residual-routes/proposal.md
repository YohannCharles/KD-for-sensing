## Why

GPS coarse anchor、Top8 selector 训练、GPS prior 后接其它模态 residual/delta correction 这组研究路线已经被实验结果证明精度过低且差值学习困难，继续保留会扩大维护面、误导后续论文路线，并让当前 README、CLI、配置和测试矩阵继续围绕失败假设增长。BGAM 模块按用户最新指令保留。

本次变更将这些已放弃路线从当前源码支持面中直接退役；用户已明确“不管兼容”，因此不保留旧 console script、配置 alias 或迁移包装层。

## What Changes

- **BREAKING** 删除 DeepSense6G Top8 selector 训练/plot/compare、standalone MMW Top8 candidate manifest 入口、DeepSense6G residual fusion、camera residual、GPS coarse anchor 和 geometry residual delta 相关当前可运行入口。
- **BREAKING** 删除上述路线对应的配置、CLI、engine、data manifest/dataset、model、loss、plot/compare 脚本和 focused tests，不新增兼容 alias、stub CLI 或 registry fallback。
- **BREAKING** 从 `pyproject.toml` 移除已退役 console scripts；安装后的命令集合不再包含 retired top8 selector、residual、camera-residual 或 gps-coarse-anchor 相关入口。
- 更新 README、README_REPRODUCE、docs/inventory、OpenSpec 当前 specs 和架构 guardrail，把这些路线标记为已放弃并从 quickstart/当前推荐 workflow 中移除。
- 保留非该失败路线的 GPS-only 与主线能力，例如 GPS-Rel-Polar 预处理/模型、DeepSense6G GPS v2、MMW Town GPS v2、DeepSense6G/MMW GPS+LiDAR BGAM、GPS window baseline、JEPA GPS conditioning、CSI hardening、Raymobtime、viewer manifest、BGAM 依赖的 TopK candidate manifest/loss 支撑代码和通用 circular metrics。
- 不自动删除 `outputs/`、`logs/`、cache、checkpoint、dataset 或历史权重；如后续需要清理这些本地产物，必须走 runtime cleanup manifest 或用户单独显式确认。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-architecture`: 增加“用户明确退役失败实验路线时，源码支持面必须直接删除且不保留兼容入口”的架构约束。
- `experiment-workflow`: 从当前推荐工作流中移除 retired Top8 selector/residual/GPS coarse anchor 路线，并要求 README/CLI/inventory 与保留主线一致。
- `gps-coarse-anchor-prediction`: 将 GPS coarse anchor、residual preview、GPS prior fallback 和 pseudo-history 导出能力标记为已退役。
- `deepsense6g-gps-residual-fusion`: 将 DeepSense6G GPS prior anchored residual correction workflow 标记为已退役。
- `deepsense6g-camera-ae-residual-correction`: 将 camera-assisted residual correction workflow 标记为已退役，同时保留独立 Camera AE 如仍被其它当前路线使用。
- `deepsense6g-gps-top8-candidate-selector`: 将 DeepSense6G GPS Top8 selector 训练/plot/compare workflow 标记为已退役；BGAM 依赖的 TopK candidate manifest 支撑代码保留。
- `mmw-town-gps-top8-candidate-selector`: 将 standalone MMW Town GPS Top8 candidate manifest CLI/config 标记为已退役；BGAM 内部候选 manifest 支撑代码保留。
- `geometry-residual-beam-labels`: 将专门服务 residual/delta 失败路线的 geometry residual label 能力标记为已退役。

## Impact

- 影响入口：`pyproject.toml` 中 retired top8 selector、residual、camera residual 和 gps coarse anchor console scripts；BGAM console scripts 保留。
- 影响源码：`src/kd_sensing/cli/`、`data/`、`engine/`、`models/`、`losses/` 中上述路线专属模块及其导出/注册引用。
- 影响配置和脚本：`configs/*top8*`、`configs/*residual*`、`configs/gps/*coarse*` 及相关 run/analysis shell 或 Python orchestration；`configs/*bgam*` 保留。
- 影响测试：删除或改写专门覆盖退役路线的 focused tests；保留架构、GPS v2、基础 GPS、JEPA、CSI、Raymobtime、viewer 等当前主线测试。
- 影响文档和规范：README、README_REPRODUCE、docs/project surface inventory、OpenSpec 当前 specs 和归档说明需要同步，避免旧失败路线继续显示为可运行主线。
