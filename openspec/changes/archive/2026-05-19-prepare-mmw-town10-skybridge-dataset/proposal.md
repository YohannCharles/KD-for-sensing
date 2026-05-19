## Why

MMW 的 Town10 skybridge 场景同时提供传感器数据和 V2I 信道数据，但项目当前只有 DeepSense6G 的预处理与训练入口，无法直接把 `Town10_skybridge_seed24.zip` 和信道 `Town10.zip` 转成项目可消费的多模态样本与 beam 标签。需要新增可复现的数据准备流程，将 MMW 的 CARLA 传感器帧、信道路径和派生 beam 标签落到统一目录与 manifest 中，为后续多模态 beam 预测实验提供输入。

## What Changes

- 新增 MMW Town10 skybridge 数据准备入口，接受本地 `Town10_skybridge_seed24.zip` 和信道 `Town10.zip` 路径，按项目 MMW 目录约定解包到 `dataset/MMW/<condition>/Sensor_Data` 与 `dataset/MMW/<condition>/Channel_Data`。
- 参考 MMW 官方 V2I 文件层级，发现并校验 `Town10/Town10_skybridge_seed24/<agent>/<frame>` 下按六位帧号同步的 CAV/RSU 模态文件。
- 为每个有效帧或序列生成 manifest/CSV，记录 RGB camera、LiDAR、RSU radar、GPS/IMU、RSU/CAV metadata、信道 `_paths.npy`、历史 beam 和 future beam 标签路径。
- 从信道多径数据派生固定 beam codebook 下的 beam power vector 与 top-1 beam 标签，输出可被现有 beam 预测任务复用的标签文件。
- 写出 split、metadata 和 sanity report，记录输入 zip、场景、agent、帧覆盖、跳过原因、beam 分布和输出路径，保证处理结果可审计与复现。
- 提供轻量测试夹具覆盖解包索引、帧对齐、信道到 beam 标签派生、manifest 字段和缺失文件诊断。
- 不在本变更中下载 MMW 数据，不迁移本地大 zip，不训练模型。

## Capabilities

### New Capabilities
- `mmw-town10-dataset-preparation`: 定义 MMW Town10 skybridge 传感器 zip 与信道 zip 的解包、模态索引、信道派生 beam 标签、manifest/split/metadata 产出和质量检查契约。

### Modified Capabilities
- `modality-aware-data-loading`: 数据加载契约需要接受由 MMW 准备流程生成的 manifest/CSV，并在启用对应模态时按需读取 MMW 派生的 image、LiDAR、GPS、mmWave/channel feature 与 beam 标签字段。
- `mmwave-preprocessing`: mmWave 预处理契约需要支持从 MMW 信道 `_paths.npy` 派生 64-beam power vector，而不只读取 DeepSense6G 的现成 power txt。

## Impact

- 可能新增 `src/kd_sensing/data/mmw/` 或等价模块，承载 zip 解包索引、MMW metadata 解析、channel-to-beam 派生、manifest/split/sanity 写出。
- 可能新增 `scripts/mmw/prepare_town10_skybridge.py` 或统一 CLI 子命令，并新增 `configs/preprocess/mmw_town10_skybridge.yaml`。
- 影响 `src/kd_sensing/data/layouts.py` 的 MMW condition/root helper，以及 dataset builder 对 MMW manifest 的识别。
- 影响现有 `mmwave` 输入约定：保留 DeepSense6G 64 维 power vector 读取，同时新增 MMW channel-derived 64 维 power vector 生成路径。
- 新增测试文件，例如 `tests/test_mmw_town10_preparation.py`，并补充数据加载相关回归测试。
- 项目相关 Python 命令、测试和 smoke 验证均使用 `conda run -n kd_mm_beam ...`。
