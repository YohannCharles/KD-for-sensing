## Why

当前模型注册表同时暴露了 `modular_sequence` 组件化主路径、旧 strong/lightweight 整模型、旧 feature extractor 整模型注册和若干别名，导致查看 `MODELS.list()` 或模型架构清单时难以判断哪些是当前推荐入口、哪些只是历史兼容残留。现在已有 `model_architecture_inventory.md` 和架构摘要 CLI，可以支撑一次有审计边界的收口：保留 current encoder/core/head 能力，把普通 baseline 统一迁移到 `modular_sequence`，并把旧注册名改为可诊断的 removed guard。

## What Changes

- **BREAKING** 退役第一批无当前配置依赖或明显重复的 registry surface：
  - `MODELS`: `modular_sequence_model`、`gps_only_neural_baseline`、`radar_feature_extractor`、`lidar_feature_extractor`、`mmwave_feature_extractor`。
  - `REPRESENTATION_CORES`: `jepa_token_transformer` 别名。
  - `HEADS`: `safe_residual_reranker` 别名。
  - `ENCODERS`: `point_cloud_mlp`，除非实现阶段发现仍有未登记 current workflow 依赖；若保留，必须降级为 experimental/hidden 文档状态而非 current 推荐入口。
- **BREAKING** 迁移第二批旧整模型后退役：
  - 单模态旧整模型：`image_strong`、`image_lightweight`、`radar_strong`、`radar_lightweight`、`gps_strong`、`gps_lightweight`、`lidar_strong`、`lidar_lightweight`、`mmwave_strong`、`mmwave_lightweight`。
  - Fusion 旧整模型：`fusion_lightweight`；`fusion_strong` 若确认无 current config/test/spec 语义依赖，也一并退役为 removed guard。
- 将受影响的 canonical/root configs 迁移到 `model.primary.type: modular_sequence`：
  - Radar：`radar_cnn + single_gru + beam_head`。
  - GPS：`gps_mlp + single_gru + beam_head`。
  - mmWave：`mmwave_mlp + single_gru + beam_head`。
  - Fusion radar+GPS：`radar_cnn + gps_mlp + early_concat_gru + beam_head`。
  - Image 与 LiDAR canonical configs 已基本使用 `modular_sequence`，本 change 只清理残留旧注册名和测试/文档期待。
- 保留当前明确能力，不纳入本次退役：
  - `modular_sequence`、`resnet18_imagenet_rgb`、TinyViT 四个 encoder、`jepa_context_image`、`camera_ae_frozen`、`gps_conditioned_jepa`、`jepa_msac`、`bev_fusion_2604`、`vision_position_late_fusion`、`cls_token_transformer_fusion`、`token_transformer_fusion`、`pilot_dual_view_csi`、`gps_mlp`、`radar_cnn`、`lidar_cnn`、`mmwave_mlp`、主要 projector/core/head。
- 更新 registry allowlist、architecture boundary tests、模型架构目录、扩展指南、维护上下文索引和相关 OpenSpec specs，使当前推荐面只展示组件化路径和显式 workflow/paper exception。
- 为所有退役名称添加清晰 `register_removed(...)` guard，错误信息必须给出迁移目标，例如 `radar_strong -> modular_sequence + radar_cnn + single_gru`。

## Capabilities

### New Capabilities

- 无。本 change 不新增模型能力，只收缩现有模型 registry surface。

### Modified Capabilities

- `component-registry`: 注册表可发现列表、removed guard 和默认组件导入边界需要反映 legacy model/encoder/core/head 名称退役。
- `model-architecture-extension-contract`: 普通 supervised/adaptation baseline 的整模型注册退役规则和 `modular_sequence` 迁移要求需要成为明确契约。
- `modular-sequence-model`: 更新“保留 legacy 注册名”语义，允许 canonical 单模态和轻量 fusion root config 迁移到 `modular_sequence` 后将旧整模型名称退役。
- `gps-modality-model`: GPS canonical strong/lightweight/supervised 配置改为 `modular_sequence + gps_mlp`，旧 GPS strong/lightweight/teacher/student 注册名只保留 removed guard。
- `radar-teacher-model`: Radar strong/supervised 配置改为 `modular_sequence + radar_cnn`，旧 radar strong/teacher 注册名只保留 removed guard。
- `radar-student-model`: Radar lightweight 配置改为 `modular_sequence + radar_cnn`，旧 radar lightweight/student 注册名只保留 removed guard。
- `lidar-modality-model`: LiDAR canonical 配置继续使用 `modular_sequence + lidar_cnn`，并将旧 LiDAR strong/lightweight/feature-extractor `MODELS` 注册期望收口为 removed guard 或直接类导入。
- `mmwave-modality-model`: mmWave canonical strong/lightweight/supervised 配置改为 `modular_sequence + mmwave_mlp`，旧 mmWave strong/lightweight/feature-extractor `MODELS` 注册名只保留 removed guard。
- `configurable-multimodal-fusion`: 旧 `fusion_lightweight` radar+GPS config 迁移到 `modular_sequence` fusion；如同时退役 `fusion_strong`，需在该能力中记录 fusion whole-model legacy route 的 removed guard。

## Impact

- 受影响代码：
  - `src/kd_sensing/models/{image,radar,gps,lidar,mmwave}.py`
  - `src/kd_sensing/models/fusion/networks.py`
  - `src/kd_sensing/models/modular.py`
  - `src/kd_sensing/registries.py`
  - `src/kd_sensing/models/architecture_summary.py` 如需过滤/标注 removed 名称
- 受影响配置：
  - `configs/radar/{strong,lightweight,supervised}.yaml`
  - `configs/gps/{strong,lightweight,supervised,ablation_relative_polar}.yaml`
  - `configs/mmwave/{strong,lightweight,supervised}.yaml`
  - `configs/fusion/radar_gps_supervised.yaml`
  - Image/LiDAR config 只做残留字段、命名和文档一致性检查。
- 受影响测试与治理：
  - `tests/test_config_load_characterization.py`
  - `tests/test_architecture_boundaries.py`
  - 模型 focused tests：radar/GPS/LiDAR/mmWave/modular sequence/registry removed guard。
  - `docs/maintainer_context_index.yaml` 的 model allowlist。
  - `docs/model_architecture_inventory.md`、`docs/project_surface_inventory.md`、`docs/extension_guide.md`。
- 兼容性：
  - 使用旧注册名的本地配置会失败，并收到迁移目标提示。
  - 当前 root/canonical configs 必须继续可加载、可构建并通过 synthetic forward/config tests。
  - 不读取真实 `dataset/`，不生成训练输出、cache、logs 或 checkpoint。
