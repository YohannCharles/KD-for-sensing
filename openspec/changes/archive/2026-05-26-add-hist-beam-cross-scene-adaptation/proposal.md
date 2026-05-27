## Why

DeepSense6G 31-34 的跨场景 beam prediction 是当前项目从单场景训练走向可迁移验证的自然下一步；现有框架已经具备多模态 fusion、ResNet-18 图像 encoder、GPS/Radar 数据路径和配置驱动训练，但缺少 leave-one-scene-out、target adaptation、层次化 beam label 与 shared/private 解耦的统一实验能力。

本变更将《跨场景自适应方案.md》收敛为一套快速验证版 HiST-Beam 工作流，用 2-4 周验证“coarse beam semantics + scene-private refinement + lightweight target adapter”是否值得继续扩展，而不是一次性引入完整投稿系统。

## What Changes

- 新增 HiST-Beam 快速验证模型能力：
  - 支持 flat、hierarchical、shared-private、decoupled、adapter-only、adapter+prototype 和 full fine-tuning 变体。
  - 在现有 `experiment.task: fusion`、注册表、batch preparation、ModelOutput 和指标流程上接入，不新增旧式根目录训练脚本。
  - 首阶段固定使用 `image`、`radar`、`gps` 三模态，LiDAR 仅保留后续扩展边界。
- 新增跨场景 LOSO 实验工作流：
  - 支持 DeepSense6G scenarios 31-34 的 4 折 source/target 实验。
  - target scene 内部拆分为 `target_adapt` 和 `target_test`，并确保 `target_test` 不参与训练、适应、阈值选择或 prototype 选择。
  - 支持 few-shot target label budgets `0/5/10/20/50` 与 seed 矩阵，优先 coarse-group stratified sampling。
- 新增层次化 beam label、解耦 loss、prototype 生成/加载、target adapter 冻结策略和 adaptation 指标/产物。
- 新增配置和 orchestration 入口，复用包内 CLI/模块入口，输出 source/adapt/test metrics、checkpoint、predictions、prototypes 和矩阵汇总。
- 修改 DeepSense6G 场景选择契约，将 Scenario 33 和 Scenario 34 纳入规范场景注册和默认路径解析。
- 不引入本地数据、训练输出、日志、cache 或新 checkpoint 到源码变更。

## Capabilities

### New Capabilities
- `hist-beam-cross-scene-adaptation`: 定义 HiST-Beam 模型变体、层次化 beam 输出、shared/private 解耦、target adapter、prototype alignment、adaptation 指标与运行产物契约。
- `cross-scene-loso-workflow`: 定义 DeepSense6G 31-34 的 leave-one-scene-out fold、target adapt/test split、few-shot sampling、运行编排和结果汇总契约。

### Modified Capabilities
- `deepsense6g-scene-selection`: 将 DeepSense6G Scenario 33 和 Scenario 34 加入受支持场景、别名、默认数据根目录和 metadata 记录规则。

## Impact

- 代码：
  - `src/kd_sensing/data/scenes.py` 和相关场景测试需要支持 Scenario 33/34。
  - `src/kd_sensing/engine/data_factory.py` 或新的窄模块需要构建 source multi-scene、target adapt、target test dataloader，并复用训练集 normalizer/scaler artifact。
  - `src/kd_sensing/models/` 需要新增注册式 HiST-Beam fusion 模型、hierarchical heads、adapter 和 GRL/helper。
  - `src/kd_sensing/engine/` 与 `src/kd_sensing/evaluation/` 需要接入 hierarchical loss、decoupling/prototype loss、coarse/fine metrics、trainable parameter ratio、adaptation time 和 predictions 导出。
  - `src/kd_sensing/cli/` 或包内模块需要提供配置驱动 LOSO/adaptation orchestration，保持现有 CLI 边界。
- 配置：
  - 新增 `configs/hist_beam/` 或等价配置目录，提供 base、V0-V6 variants、LOSO matrix 和 quick-smoke 配置。
  - 所有项目相关 Python 命令继续使用 `conda run -n kd_mm_beam ...`。
- 测试：
  - 增加场景解析、LOSO split 防泄漏、few-shot sampling、hierarchical label/loss、adapter 冻结、prototype artifact、metrics 和 CLI smoke 测试。
- 文档：
  - README 只补充快速入口；详细实验矩阵和设计放在 OpenSpec 或 docs 中，避免 README 承载完整研究方案。
