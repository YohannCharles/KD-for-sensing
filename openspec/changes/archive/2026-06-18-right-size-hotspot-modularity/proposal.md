## Why

当前热点治理已经能阻止超长函数、超长类和 facade 静默膨胀，但项目仍存在多个真实维护瓶颈：`data_factory.py`、`image_ae_gps.py`、DeepSense6G/MMW dataset、trainer orchestration 和 JEPA benchmark 分析模块都需要更彻底的职责收口。既然可以接受高风险，本 change 将从保守的“右尺寸化规则”升级为完整源码表面修复：该拆的拆、该合并的合并、该保留的明确接受并加测试护栏。

## What Changes

- 将热点治理升级为分阶段修复 campaign，而不仅是预算元数据调整：每个 wave 都要有 baseline、迁移边界、focused tests 和回滚点。
- 重构 `src/kd_sensing/engine/data_factory.py`：保留公开构建入口，拆出 dataloader kwargs、protocol split、stratified/group split、internal validation split、GPS scaler/normalizer 协调等职责模块。
- 重构 `src/kd_sensing/preprocessing/sequences.py`：拆出 column plan、window materialization、balanced split selection、metadata writing 和 label distribution helper，保留 `SequencePreprocessor` 为薄 orchestration。
- 重构 BeamBench Image AE+GPS baseline：把 2400+ 行 `image_ae_gps.py` 拆成 config、dataset/model、AE feature cache、training/evaluation、paper split orchestration、report writers、torch runtime helpers；公开 import/CLI 语义保持稳定。
- 重构 DeepSense6G/MMW dataset：继续从 dataset 类中抽出 sample assembly、resource reader glue、scaler/normalizer setup、target provider adapter、MMW derived-column/geometry/radio semantic helpers。
- 重构 `trainer._train_inner`：拆出 dataloader setup、training runtime plan、epoch loop、validation/checkpoint coordination、final evaluation 和 artifact finalization。
- 重构 JEPA benchmark 第二层模块：`jepa_benchmark_common.py` 拆成 schema/scalar/metadata/io helpers；`jepa_benchmark_scenario_d.py` 拆成 suite normalization、CxD phase、dominance/crossing、failure modes 和 metric rows；runner 拆出 summary/bundle/manifest writer。
- 明确不拆的对象：`losses/jepa.py`、`losses/gps_lidar_bgam_losses.py` 和 `models/csi_encoder.py` 当前规模较小且领域内聚，默认保留；只有发现重复抽象、跨模块复用或测试缺口时才调整。
- 合并低价值边界：单调用点包装类、只为降行数产生的 helper、重复 `utils` 聚合、无公开兼容价值的 facade 应合并回清晰 owner 或改为私有局部 helper。
- 更新架构边界测试、维护上下文索引、inventory 和 AI 导航，确保后续 agent 能理解每个热点的 owner、wave、enforcement、split/consolidation target 和验证命令。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-health-guardrails`: 将维护性热点检查从单一预算检查扩展为分阶段源码表面修复护栏，覆盖高风险 wave、baseline capture、hard facade budget、merge candidate 和 accepted-size。
- `maintainer-context-index`: 扩展 hotspot metadata，使索引能记录 remediation wave、owner、enforcement、headroom、split/consolidation targets、accepted-size rationale 和 focused validation。
- `ai-maintainer-navigation`: 更新非平凡改动前的 AI 维护导航，要求 agent 按完整修复计划判断拆分、合并、保留、分阶段验证和回滚边界。

## Impact

- 影响源码结构：`src/kd_sensing/engine/`、`src/kd_sensing/preprocessing/`、`src/kd_sensing/baselines/beambench/`、`src/kd_sensing/data/datasets/`、`src/kd_sensing/diagnostics/` 和少量 import 调整。
- 影响治理文档与测试：`docs/maintainer_context_index.yaml`、`docs/project_surface_inventory.md`、`docs/agent_navigation.md`、`tests/test_architecture_boundaries.py` 和相关 focused tests。
- 公开 API/CLI 目标：包内推荐入口、console scripts 和已登记 public import 语义必须保持；内部 helper import 可迁移到新 owner 模块。
- 风险：这是高风险结构重构，可能触发 import drift、测试覆盖缺口、循环依赖或行为回归；实施必须按 wave 合并，每个 wave 单独验证。
- 不改变目标：训练数学语义、数据 split 语义、beam label 语义、checkpoint schema、默认输出目录、本地产物边界和退役路线拒绝边界必须保持稳定。
