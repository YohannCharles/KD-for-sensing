## Why

`image_motion` 路径依赖相邻帧差分 mask 和专用缓存，适用面较窄，并且持续污染 image 模态、cache policy、诊断和模型注册的主路径。现在将 image 输入统一收敛到 RGB/ImageNet 路径，可以减少维护面，避免继续为低价值 legacy 分支保留兼容。

## What Changes

- **BREAKING** 删除所有 image motion mask 数据转换、cache key、cache 读写、预热入口、配置字段和运行 metadata 字段。
- **BREAKING** 删除 `motion_mask` image profile、`motion_cnn`/`legacy_motion_cnn` image encoder 注册名，以及依赖单通道 motion mask 的 legacy image/fusion 模型路径。
- **BREAKING** 现有 image-only、image+radar fusion、CRAF、MARF 或诊断配置不得再隐式回退到 image motion；包含 image 的配置必须使用 RGB/ImageNet image 输入和兼容的三通道 encoder。
- 保留已有 `outputs/` 目录和历史实验产物，不迁移、不清理、不保证旧 checkpoint 可继续加载。
- 更新文档、OpenSpec specs、测试和示例配置，移除所有鼓励预热或复用 image motion cache 的说明。

## Capabilities

### New Capabilities
无。

### Modified Capabilities
- `automated-cache-policy`: 自动 cache policy 不再管理 image motion cache，仅保留仍受支持的 cache 类型，例如 LiDAR BEV。
- `modality-aware-data-loading`: image 模态加载不再支持 motion mask 或 image motion cache，非 image 任务也不再携带 image cache 分支。
- `training-throughput-optimization`: 删除 image motion mask cache 的吞吐优化需求和预处理入口。
- `modality-visual-diagnostics`: viewer manifest 只展示当前 RGB/ImageNet image 输入，不再引用 processed image motion mask 或读取 image motion cache。
- `experiment-workflow`: 训练、评估和 profile 入口不再解析或记录 `image_motion_*` 低层开关。
- `original-code-compatibility`: 取消 legacy motion mask image 分支和相关 checkpoint/config 兼容承诺。
- `project-architecture`: 数据转换、兼容 facade 和产物边界不再包含 image motion 实现。
- `modality-contracts`: image 模态契约不再暴露 motion profile、motion cache 能力或 motion encoder 推荐。

## Impact

- 影响代码：`src/kd_sensing/data/transform_ops/image.py`、`src/kd_sensing/data/transform_ops/cache.py`、`src/kd_sensing/data/transform_ops/_legacy.py`、`src/kd_sensing/preprocessing/image.py`、`src/kd_sensing/data/datasets/scenario9.py`、`src/kd_sensing/engine/cache_policy.py`、`src/kd_sensing/engine/run_metadata.py`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/config/*`、`src/kd_sensing/models/image_encoders.py`、`src/kd_sensing/models/modular.py`、诊断 manifest/prediction 相关模块和 CLI 预处理入口。
- 影响配置：删除 `configs/preprocess/image_motion_cache.yaml` 和所有 `image_motion_*` 字段；包含 image 的训练配置必须落到 RGB/ImageNet 路径。
- 影响测试：删除或重写 image motion cache、motion profile、legacy motion encoder 和兼容 checkpoint 相关测试；新增 RGB image 路径和无 image motion 引用的回归检查。
- 影响文档：README、吞吐文档、可视化说明和 OpenSpec active change `add-resnet18-image-architecture` 中所有 motion mask 兼容描述都需要同步移除。
- 不影响：历史 `outputs/` 目录不删除；LiDAR BEV cache、beam label cache、radar/GPS/mmWave/LiDAR 数据处理保持独立可用。
