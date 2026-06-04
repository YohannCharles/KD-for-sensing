## ADDED Requirements

### Requirement: GPS+LiDAR BGAM 包内入口
项目 MUST 将 GPS+LiDAR BGAM reranker 的实现放入 `src/kd_sensing/` 包内。manifest enrich、dataset、geometry utility、model、loss、engine、evaluation、debug plot 和 CLI MUST 按现有职责边界分布在 `kd_sensing.utils`、`kd_sensing.data`、`kd_sensing.models`、`kd_sensing.losses`、`kd_sensing.engine`、`kd_sensing.evaluation` 和 `kd_sensing.cli` 中。项目 MUST NOT 新增长期维护的顶层 `train_gps_lidar_bgam.py`、`eval_gps_lidar_bgam.py`、`datasets/gps_lidar_dataset.py` 或 `models/gps_lidar_bgam.py` 旁路入口。

#### Scenario: console scripts 暴露 BGAM workflow
- **WHEN** 开发者完成 editable install 并查看 `pyproject.toml` entry points
- **THEN** 项目 MUST 暴露 GPS+LiDAR BGAM 相关 console scripts
- **AND** scripts MUST 至少覆盖 manifest enrich、训练/评估运行和独立评估
- **AND** 每个 console script MUST 委托 `kd_sensing.cli.*` 中的包内实现

#### Scenario: 包内 module CLI 可运行
- **WHEN** 用户执行 `conda run -n kd_mm_beam python -m kd_sensing.cli.run_deepsense6g_gps_lidar_bgam --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 `--config`、`--support-ratio`、`--label-space`、`--topk` 和 checkpoint 或 evaluation 相关参数

#### Scenario: 不新增顶层旧入口
- **WHEN** 架构边界测试扫描新 workflow
- **THEN** 测试 MUST 验证仓库根目录不存在新增的 `train_gps_lidar_bgam.py` 或 `eval_gps_lidar_bgam.py`
- **AND** 内部代码 MUST 不依赖顶层 `datasets.*`、`models.*` 或 `src.run_*` 入口

#### Scenario: 轻量导入边界保持稳定
- **WHEN** 开发者执行 `import kd_sensing` 或导入配置/路径轻量模块
- **THEN** 系统 MUST 不因 BGAM workflow eager import torch dataset、LiDAR point cloud reader、matplotlib plotter 或训练 runtime
- **AND** BGAM 重依赖模块 MUST 只在对应 CLI、engine 或显式模块导入时加载
