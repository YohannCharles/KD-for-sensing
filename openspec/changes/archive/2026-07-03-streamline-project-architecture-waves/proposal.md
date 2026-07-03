## Why

项目已经完成多轮去旧入口、组件化和热点拆分治理，但新增实验、模态、baseline 和诊断时仍会反复触碰 `DeepSense6GDataset` / `MMWDataset`、`ModularSequenceModel.forward`、训练主循环、benchmark runner、配置矩阵和脚本表面等高维护成本区域。当前 OpenSpec 与架构边界测试仍健康，正适合在进一步增加功能前做一次有计划的全盘结构收敛，降低后续扩展的改动半径和历史负担。

## What Changes

- 建立一轮完整的 architecture streamlining wave campaign，覆盖工作树收口、数据层组合化、训练/evaluation runtime 分层、模型 forward 阶段化、诊断 runner 收敛、配置/脚本表面瘦身、OpenSpec tombstone 折叠和健康护栏右尺寸化。
- 将 `DeepSense6GDataset` 和 `MMWDataset` 从“巨型继承连接板”收敛为 sample contract、modality resource readers、target providers、dataset adapter 和 dataset-family extension 的组合式结构；新增模态、target 或 metadata 时优先新增窄 adapter/helper，而不是继续扩大 dataset 主体。
- 将训练入口保持为 `kd-sensing-train` / `kd_sensing.engine.trainer.train`，但内部拆分为可审计 setup phases、run context、epoch/checkpoint/finalization 协调层；训练数学语义、checkpoint schema、默认输出目录和 run metadata 字段保持兼容。
- 将 `ModularSequenceModel.forward` 阶段化为输入收集、encoder dependency resolution、projection、core input assembly、head/post-processing、diagnostics assembly 和 auxiliary output assembly；普通 baseline 仍通过 `modular_sequence` 和组件 registry 扩展。
- 将 diagnostics 公开入口保持稳定，同时把 JEPA benchmark、JEPA visual analysis、MMW GPS v2、run index、cleanup/organize manifest 中的 suite-specific schema、aggregation、artifact writing 和 plotting 逻辑收敛到职责明确的窄模块。
- 将 `configs/scene31`、root fusion YAML、diagnostics manifest 和 local/manual scripts 按 lifecycle 分类收敛；可生成或本地队列型配置优先迁到 recipe/manifest/generator，避免实体 YAML 和一次性脚本继续膨胀。
- 折叠或归档不再提供当前 guard 价值的 retired-tombstone specs；仍提供 registry/config/CLI/document wording 防回流价值的 tombstone 保留，但必须在 inventory 中记录 guard 价值。
- 更新 architecture boundary tests，使它们继续拒绝旧入口回流、tracked runtime artifact、重依赖 barrel 和 facade 回流，同时避免复制完整源码目录、完整 OpenSpec prose 或大型 allowlist 镜像。
- **BREAKING**：未在 README、pyproject console scripts、current specs 或 inventory 明确登记为 public surface 的内部 import path、低价值 facade、thin wrapper、只 re-export 的 `__all__`、一次性 helper 和退役 direct import 可被删除或合并；内部调用方必须改用真实 owner 模块。
- **不改变**：当前 package CLI 名称、current canonical config 语义、dataset split 语义、beam label / label-space 口径、metric schema、checkpoint 读取边界、默认本地产物分区和已登记 current workflow 的用户可见行为。

## Capabilities

### New Capabilities

无。本 change 是现有项目架构、热点治理、数据/runtime/诊断/配置生命周期能力的系统性收敛，不引入新的用户级算法或训练能力。

### Modified Capabilities

- `project-architecture`: 明确 architecture streamlining wave 的全仓结构收敛边界、公开行为保持策略和内部 breaking import 删除边界。
- `project-hotspot-governance`: 将现有 remediation wave 从“登记/局部拆分”升级为必须执行的完整重构 campaign，并补充 wave baseline、rollback、验收和禁止混改规则。
- `dataset-runtime-contracts`: 增加组合式 dataset owner、modality reader、target provider、sample assembly 和 dataset-family adapter 的长期契约。
- `training-evaluation-runtime`: 增加训练/evaluation runtime 阶段化、run context、shared evaluation pass 拆分和 checkpoint/finalization 协调边界。
- `modular-sequence-model`: 增加 forward 阶段化、metadata/diagnostics assembly 边界和新增组件不扩大主 forward 的要求。
- `model-architecture-extension-contract`: 强化新增 baseline 对 dataset/batch/runtime/forward 分支的改动限制，并要求新组件能通过阶段化 forward 和架构摘要审计。
- `project-import-surface-consolidation`: 扩大内部 import surface 收缩范围，允许删除未登记 public facade、thin wrapper、低价值 `__all__` 和单调用点 helper。
- `project-entrypoint-lifecycle`: 增加 local/manual script 与 generated config recipe 的 lifecycle 收敛要求，防止新脚本复制 package CLI 或长期训练入口。
- `canonical-config-resolution`: 增加实体 YAML、virtual config、recipe/generated config 和 local/manual overlay 的边界，限制 Scene31 与 fusion 配置实体膨胀。
- `project-health-guardrails`: 调整架构边界测试职责，保留结构性失败，删除重复治理镜像，并新增 wave 完成后的分层验收要求。
- `spec-lifecycle-boundaries`: 增加 retired tombstone 折叠/归档判定、guard 价值记录和已完成 active change 收口前置要求。

## Impact

- 主要影响 `src/kd_sensing/data/`、`src/kd_sensing/engine/`、`src/kd_sensing/models/modular.py`、`src/kd_sensing/diagnostics/`、`src/kd_sensing/config/`、`src/kd_sensing/cli/`、`scripts/`、`configs/`、`docs/project_surface_inventory.md`、`docs/agent_navigation.md`、OpenSpec specs 和 architecture/focused tests。
- 需要在实施前收口当前工作树：归档或明确暂缓已完成 active changes，恢复或确认 `dataset/.gitkeep`，隔离未跟踪实验配置/脚本和本地 cache 噪声，记录 baseline validation。
- 验证以 wave 为单位执行：每个 wave 运行对应 focused tests 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`；最终运行 `openspec validate streamline-project-architecture-waves --strict`、`openspec validate --all --strict` 和 `conda run -n kd_mm_beam pytest -q`。
- 不删除、移动、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event、`All_models/` 历史权重或其它本地运行产物；源码表面清理只影响 tracked source、tests、configs、docs、scripts 和 OpenSpec artifacts。
