## Why

当前项目同时存在根目录 `cache/`、`outputs/cache/`、数据集目录内的 `image_derived_cache` / `lidar_bev_cache`，以及各配置、预处理入口对 cache 路径的不同默认值。真实数据、可再生成缓存、训练输出和诊断产物混放在多个位置，会增加运行前检查、清理 manifest、迁移实验和排查路径问题的成本。

本 change 目标是收敛“新生成缓存”的默认落点：`dataset/` 继续只承载原始数据、下载文件和必要 prepared 数据；可再生成缓存默认进入 `outputs/cache/` 下的语义子目录；训练、评估和诊断输出继续位于 `outputs/` 或 `logs/`。旧路径仍通过显式配置兼容，避免自动搬动大体量本地数据。

## What Changes

- 新增集中式 runtime cache layout helper，统一描述 `outputs/cache/<dataset-family>/<scope>/<cache-kind>` 路径。
- 将 DeepSense6G image-derived cache、DeepSense6G LiDAR BEV cache、MMW image/LiDAR cache 和 MMW physical label cache 的默认路径收敛到 `outputs/cache/`。
- 更新预处理配置、README 和清理/诊断说明，使新推荐命令不再默认在 `dataset/` 或根目录生成 cache。
- 保留显式旧 cache 路径兼容：用户传入 `dataset/.../image_derived_cache`、`dataset/.../lidar_bev_cache` 或 `cache/...` 时仍按配置使用。
- 只迁移低风险根目录 `cache/physical_labels` 到 `outputs/cache/physical_labels`；不自动移动 254G 级 `dataset/` 数据或历史实验输出。

## Capabilities

### Modified Capabilities

- `dataset-directory-layout`: 明确 dataset 目录不应作为新缓存默认落点，并定义 runtime cache layout。
- `project-architecture`: 扩展本地产物边界，要求默认 cache 路径集中到 `outputs/cache/`，同时保持旧本地产物不被自动删除或迁移。
- `beamspace-physical-labels`: 将 MMW physical label cache 默认根从 legacy `cache/physical_labels` 收敛到 `outputs/cache/physical_labels`。

## Impact

- 影响代码：预计涉及 `src/kd_sensing/data/layouts.py`、DeepSense6G/MMW dataset cache 解析、MMW physical label cache 默认值、LiDAR/Image cache 预处理默认值、runtime cleanup 保护/分类和相关测试。
- 与 `remove-raymobtime-s008` 的关系：Raymobtime s008 已由退役 change 删除，本 change 不再定义其 cache 默认路径。
- 影响配置：更新 `configs/preprocess/*cache*.yaml`、MMW/DeepSense6G 相关 cache 配置，避免新运行继续写入 `dataset/*/*cache` 或根 `cache/`。
- 影响文档：README 和吞吐文档说明新的 cache layout、旧路径兼容和迁移边界。
- 不影响真实数据：不自动移动、删除、压缩 `dataset/`、`outputs/`、`logs/`、checkpoint 或大型历史缓存；用户需要迁移大缓存时应通过显式命令或 cleanup manifest 审核。
