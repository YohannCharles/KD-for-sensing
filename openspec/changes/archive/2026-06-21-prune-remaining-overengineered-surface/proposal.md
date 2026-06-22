## Why

上一轮项目表面瘦身已经删掉一批旧入口和依赖，但仓库仍保留了几类“为了维护治理本身而维护”的复杂度：大量实体实验 YAML、超长架构边界测试、退役 guard 表、薄 re-export facade、只服务一个调用点的小类型文件和无调用自检 helper。它们继续增加同步成本，却不改善当前训练、评估、预处理或诊断工作流。

本变更把剩余 ponytail 审计结果转成可验证的二阶段收敛方案：先收缩规格和公开承诺，再按风险 wave 删除或合并低价值表面，避免直接删代码造成 OpenSpec 与实现互相打架。

## What Changes

- **BREAKING** 收缩仍无必要的兼容导入面：删除或停止承诺 `kd_sensing.engine.objective_metadata`、`kd_sensing.data` / `kd_sensing.data.datasets` lazy re-export、`kd_sensing.models.fusion` 旧类名迁移 facade、BeamBench `image_ae_gps` 大聚合 owner 等仅转发层；内部代码和测试迁到真实 owner 模块。
- **BREAKING** 继续降级低价值 retired-route runtime guards：只保留高频、仍有迁移价值的 KD、image profile、scene dataset alias 等明确错误；完全退役且已有 OpenSpec/inventory tombstone 的历史路线回落为普通 unknown-name 错误或集中退役说明。
- **BREAKING** 删除未消费的公开 helper：移除 `registry_self_check`、`_typing.AnyConfig`、无内部构造点的 `SampleRow` 文件或把仍需类型迁入真实 owner；外部脚本若依赖这些非推荐入口需改用标准 `dict[str, Any]` 或 owner 模块路径。
- 收缩配置表面：将 JEPA image+GPS、CSI hardening、BEV-Fusion、pretraining smoke 等重复实体 YAML 分类为 canonical/root、recipe/overlay 可生成、人工样例、debug/smoke 或归档；删除可由 recipe/manifest 无损表达的实体 YAML，并保留实体 YAML 优先语义。
- 重写健康护栏：把 `tests/test_architecture_boundaries.py` 从 2000+ 行手写治理镜像缩成少量结构检查，直接验证 pyproject、真实路径、OpenSpec lifecycle、配置引用、轻量导入和本地产物边界。
- 折叠重复实现：删除第二份 `deep_merge`，统一使用 `kd_sensing.config.io.deep_merge`；禁止新增等价的小工具文件或二级聚合层。
- 简化一次性分析脚本：删除或归档 `scripts/analyze_csi_hardening_sweep.py` 这类只服务已完成调试结论的研究脚本；保留当前文档报告和必要的 focused tests。
- 清理 Python module API 风格：消除 benchmark 内部 `import *` 和不必要 `__all__` 镜像，让模块显式导入实际使用符号；保留必要公开 CLI/facade 的薄入口。
- 保持核心行为不变：不改变训练数学语义、数据 split、beam label、checkpoint schema、指标口径、默认输出分区、当前 package CLI 和本地运行产物保护边界。

## Capabilities

### New Capabilities

- 无。本变更只收缩现有项目表面和治理承诺，不引入新的用户功能。

### Modified Capabilities

- `project-surface-cleanup`: 明确剩余低价值源码、配置、脚本、guard 和治理表面的删除/合并策略。
- `project-architecture`: 调整公开 import/facade 契约，允许删除无当前 public surface 价值的 package-level re-export 和薄聚合 owner。
- `project-health-guardrails`: 将架构边界测试收缩为结构事实检查，禁止继续维护大型 prose/allowlist 镜像。
- `canonical-config-resolution`: 扩展高级配置矩阵 recipe/overlay 收敛要求，支持删除重复实体 YAML 前的候选分类和等价检查。
- `component-registry`: 进一步收缩 removed guard 和无调用 registry helper，明确 registry 只保留构建所需能力。
- `dataset-runtime-contracts`: 允许把 `SampleRow` 等轻量 row 类型迁入 target-shot owner 或完全用 Mapping 表达，删除独立未消费 runtime framework 文件。
- `beambench-baseline-reproduction`: 将 BeamBench Image AE+GPS public owner 从大聚合 re-export 收缩到 package CLI 和具体实现模块，保持训练/评估行为与输出契约。

## Impact

- 代码范围：`src/kd_sensing/config/`、`src/kd_sensing/registries.py`、`src/kd_sensing/data/`、`src/kd_sensing/engine/`、`src/kd_sensing/models/fusion/`、`src/kd_sensing/baselines/beambench/`、`src/kd_sensing/diagnostics/jepa_benchmark_*`、`scripts/analyze_csi_hardening_sweep.py` 和相关 tests。
- 配置范围：`configs/fusion/experiments/jepa_image_gps/`、`configs/csi/hardening_matrix/`、`configs/fusion/csi_hardening_matrix/`、`configs/fusion/experiments/bev_fusion_2604/`、`configs/pretraining/`、`configs/diagnostics/` 中可 recipe/manifest 化的实体 YAML。
- 文档/OpenSpec 范围：`docs/project_surface_inventory.md`、`docs/agent_navigation.md`、`docs/maintainer_context_index.yaml`、README 相关入口说明和本 change 对应 delta specs。
- 测试范围：`tests/test_architecture_boundaries.py`、config load tests、component registry tests、dataset/target-shot split tests、BeamBench focused tests、CLI help smoke。
- 依赖/API 影响：默认 runtime dependency 不新增；可能删除内部兼容 import 路径和非推荐 helper。当前 `kd-sensing-*` CLI、canonical config、训练/评估/预处理/诊断主入口保持可用。
- 产物边界：不得删除或修改 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 历史权重或本地训练产物。
