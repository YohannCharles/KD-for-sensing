## Why

Multimodal-NF 的 `fusion_all_tasks` 在进入训练前会解析 image/LiDAR 派生缓存；当前本地大量 `.npy` cache 已存在，但 sidecar 仍是旧 `multimodal_nf_derived_v1`，新版 `read_only`/轻量校验将其判定为无效，导致训练在 GPU 前失败或长时间做磁盘 IO，看起来像“没占 GPU、没动”。现在需要把旧 cache 元数据升级路径做成一等能力，避免为了补 metadata 重写 118G 级 `.npy`，并让用户能从预处理、profile 和 run metadata 中清楚看到卡在 cache 阶段还是训练阶段。

## What Changes

- 为 Multimodal-NF image/LiDAR derived cache 增加 sidecar-only 迁移/升级能力：当 `.npy` 数据文件可用且旧 sidecar 已包含足够的核心身份字段时，系统可补齐 `cache_schema_version`、source identity、IO layout、bytes、shape、dtype、recommended access pattern 等 v2 轻量校验字段，而不重写 `.npy` 数据。
- 调整 `auto`/预热路径：优先执行 metadata-only upgrade；只有数据文件缺失、shape/dtype 与源文件不一致、强校验失败或用户显式 `rebuild=true` 时才重建 `.npy`。
- 保持 `read_only` 的安全语义：默认不偷偷改写 cache；但错误信息必须明确区分“cache 数据缺失”“sidecar 可迁移但未升级”“sidecar/数据真正不匹配”，并给出可执行预处理命令。
- 增加显式预处理/校验参数或行为，用于批量升级 train/test image/LiDAR sidecar，输出 upgraded/rebuilt/skipped/failed 计数。
- 扩展 Multimodal-NF profile 和并行训练推荐，使其能报告 v1/v2 sidecar 覆盖率、migration pending 数量、cache validation mode、是否会触发重建，以及推荐的预热/升级命令。
- 增加 focused tests，覆盖旧 sidecar metadata-only upgrade、`read_only` 清晰失败、`auto` 不重写 `.npy` 的升级、强校验不匹配时拒绝升级、以及 fusion multitask 配置的启动前 cache 状态可诊断。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `multimodal-nf-dataset`: 修改 Multimodal-NF image/LiDAR 派生缓存 requirement，新增旧 sidecar 可迁移语义、metadata-only upgrade 行为和更清晰的 `read_only`/`auto` 失败或升级契约。
- `training-throughput-optimization`: 修改 Multimodal-NF 吞吐 profile 与推荐 requirement，新增 sidecar schema 覆盖率、migration pending、预热/升级建议和“训练尚未进入 GPU 阶段”的诊断字段。

## Impact

- 受影响代码：
  - `src/kd_sensing/preprocessing/multimodal_nf_derived_cache.py`
  - `src/kd_sensing/data/datasets/multimodal_nf.py`
  - `scripts/profile_training_io.py`
  - `scripts/recommend_parallel_training.py`
  - 相关 config/runtime metadata helper 和 tests
- 受影响配置：
  - `configs/preprocess/multimodal_nf_derived_cache.yaml`
  - `configs/multimodal_nf/fusion_all_tasks.yaml` 的推荐运行方式和诊断说明
- 不改变 Multimodal-NF 样本字段、target 语义、模型接口、指标口径或真实数据目录布局。
- 不提交真实 cache、训练输出、日志、checkpoint 或本地 profile 产物。
