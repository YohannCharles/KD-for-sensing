## Why

当前仓库已经形成 `ENCODERS`、`MODELS`、`modular_sequence` 和 `training_strategy_metadata()` 的模型扩展边界，但参数量、组件组合、token/compute proxy 和 sweep 候选统计仍分散在训练启动 summary、JEPA sweep manifest 估算器和个别 encoder metadata 中。随着 TinyViT、CNN/hybrid JEPA、scene-conditioned meta-offset 等候选同时推进，需要一个统一、可审计、可复用的模型架构与参数摘要能力，帮助快速比较当前正在使用的整体模型、各模态 encoder 和极小参数量 JEPA 候选。

## What Changes

- 新增模型架构摘要能力，基于已解析配置和真实 `nn.Module` 实例输出统一 summary，不新增第二套 registry，不改变模型构造语义。
- 提供通用参数统计 helper，覆盖 total/trainable/frozen 参数、按组件角色分组的参数量、未使用或语义排除参数、模块路径、组件 registry type、class 和 freeze/checkpoint metadata。
- 支持 `modular_sequence` 组件级摘要：image/GPS/LiDAR/mmWave/CSI 等 encoder、projector、representation core、head、geometry prior、reranker 和可选 whole-model exception 都使用同一输出 schema。
- 支持 JEPA visual architecture sweep 候选摘要和现有 manifest 参数口径收敛，明确纳入 `patch14_stage1_gps_query` 极小参数量基线，并与 `resnet18_layer4_tokens`、`resnet18_layer3_layer4_tokens` 等 CNN token 候选同表比较。
- 支持配置/override 预检 warning，例如 encoder 专属 `unfreeze_stages` 不兼容、潜在 checkpoint 下载、TinyViT 下游未使用分类 head 参数、manifest 估算值与真实实例统计值不一致。
- 新增薄 CLI 或等价包内入口，用于对单个配置、配置+override、sweep manifest 或已存在 `startup_summary.json` 输出 JSON/Markdown/CSV summary；默认不训练、不读取真实 dataset、不写 checkpoint。
- 训练 startup summary、TensorBoard startup scalars 和 sweep summary 可复用新 helper，但保持现有 artifact 字段向后兼容。
- **BREAKING**: 无。现有模型注册名、训练入口、评估入口、配置加载和 run artifact schema 保持兼容；新增 summary 字段为 additive。

## Capabilities

### New Capabilities

- `model-architecture-summary`: 定义模型架构与参数摘要的统一 schema、实例级参数统计、候选级 summary、warning、CLI/输出边界和与训练/sweep 的集成契约。

### Modified Capabilities

- `model-architecture-extension-contract`: 新增要求：新增 baseline、组件 baseline、whole-model exception 和 workflow/paper reproduction 必须能被架构摘要能力审计参数量、组件组合、checkpoint/freeze 策略和比较口径。
- `modular-sequence-model`: 新增要求：`modular_sequence` 必须为架构摘要暴露 encoder/projector/core/head 组件角色、registry type、metadata 和参数分组，且不改变 forward/runtime 契约。
- `jepa-visual-architecture-sweep`: 新增要求：visual sweep 的 params/compute summary 必须使用或对齐统一架构摘要 schema，并保留 `patch14_stage1_gps_query`、ResNet token 候选和 CNN/hybrid 候选的可比参数字段。

## Impact

- 影响代码区域：新增 `src/kd_sensing/models/architecture_summary.py` 或等价窄模块；复用/轻微调整 `src/kd_sensing/engine/debug_diagnostics.py` 的参数统计；必要时更新 `src/kd_sensing/models/modular.py` 的 metadata handoff；更新 `src/kd_sensing/diagnostics/cnn_hybrid_jepa_visual_prior_sweep.py` 的 params/compute summary 入口；新增包内薄 CLI 时更新 `src/kd_sensing/cli/`、`pyproject.toml` 和维护上下文索引。
- 影响配置/文档：README 只添加短索引或命令示例；详细说明放入 `docs/extension_guide.md` 或新的窄文档；如新增 CLI，需要同步 `docs/maintainer_context_index.yaml` entrypoint owner metadata 和 `docs/project_surface_inventory.md` lifecycle 说明。
- 影响测试：新增架构摘要 focused tests，覆盖 ResNet image、TinyViT scratch/22k metadata、image+GPS modular config、JEPA patch14 极小参数量候选、ResNet token 候选、sweep manifest rows、warning 和 JSON/Markdown/CSV 输出 schema；保留架构边界、配置加载和相关 sweep tests。
- 影响运行产物：summary CLI 默认写入 stdout 或用户指定 ignored `outputs/analysis/model_architecture_summary/`；不得读取真实 `dataset/`，不得访问网络或下载权重，除非用户显式构建需要下载的 pretrained 模型且允许下载。
- 依赖影响：不新增必需第三方依赖；参数统计使用 PyTorch module/parameter introspection 和现有配置/registry helper。
