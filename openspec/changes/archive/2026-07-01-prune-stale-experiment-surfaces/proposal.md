## Why

当前项目已经完成多轮去旧入口和去过度工程化，但仍保留几类维护成本高、当前运行价值低的表面：旧 CNN/hybrid JEPA full sweep、无内部调用的 LOSO dataloader helper、本地 Scene31/RBMA queue 脚本与 overlay 配置、以及少量 OpenSpec/文档对已删除入口的漂移引用。现在收口这些表面，可以减少后续 agent 和维护者误判当前支持面，也能把精力留给仍在使用的模型、数据和诊断 owner。

## What Changes

- 收敛 JEPA visual architecture sweep 表面：将当前 `jepa_visual_architecture_sweep` manifest/helper 作为推荐轻量口径，删除或降级旧 `cnn_hybrid_jepa_visual_prior_sweep` full runner；仅在仍有实际消费时保留薄兼容 reader。
- 收敛 LOSO helper 表面：确认 `src/kd_sensing/engine/loso_data.py` 的 public builder 是否仍有当前公开契约价值；若无，删除或退役该模块，同时保留 `kd_sensing.data.loso` 中的数据集无关 fold/few-shot 语义。
- 收敛本地 Scene31/RBMA 运行面：将固定 GPU shell、本地 bounded runner、`configs/scene31/`、strong-encoder 和 M2Beam 单模态 overlay 分类为 local/manual surface，删除可由统一 manifest runner 或文档命令替代的脚本/配置。
- 修复规格漂移：将仍引用已删除 `kd_sensing.cli.beambench_check_dataset` 的 BeamBench 数据检查场景改为当前 owner module 或等价包内入口；更新 inventory、README/docs/OpenSpec 引用。
- 清理低风险本地噪声：删除已被 git 忽略的 `__pycache__` 和 `.pyc`，但不触碰 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或历史权重。
- 保留核心 owner：本 change 不重构 `DeepSense6GDataset`、`MMWDataset`、`ModularSequenceModel`、训练主循环、batch contract 或模型 forward 语义；这些只在后续触碰业务语义时小步拆分。

## Capabilities

### New Capabilities

- 无。该 change 收口现有能力和当前支持面，不引入新运行能力。

### Modified Capabilities

- `project-surface-cleanup`: 增加对旧实验/诊断 surface、本地 runbook、ignored bytecode 和孤岛 helper 的删除/保留判定。
- `project-health-guardrails`: 增加对 stale current 引用、tracked 本地工具状态、facade 回流和本地脚本/config 分类漂移的检查要求。
- `jepa-visual-architecture-sweep`: 将当前轻量 architecture sweep 作为推荐口径，明确旧 CNN/hybrid full sweep 的退役或兼容降级边界。
- `cross-scene-loso-workflow`: 明确 LOSO supporting 语义不要求保留无调用的 `engine.loso_data` dataloader public surface。
- `experiment-workflow`: 明确本地 Scene31/RBMA/M2Beam queue 脚本和 overlay 配置属于 local/manual 实验面，需有删除触发条件或统一 runner 替代。
- `beambench-baseline-reproduction`: 修正 BeamBench 数据检查入口要求，移除对已删除 `kd_sensing.cli.beambench_check_dataset` 的硬编码场景。

## Impact

- 可能删除或降级的源码：`src/kd_sensing/diagnostics/cnn_hybrid_jepa_visual_prior_sweep.py`、`src/kd_sensing/engine/loso_data.py`、相关只服务这些表面的测试。
- 可能迁移或删除的本地运行入口：`scripts/run_next_v3_experiments.sh`、`scripts/run_rbma_strong_encoder_4gpu_queue.sh`、`scripts/run_m2beam_single_modal_scene31_queue.sh`、`scripts/run_rbma_missing_workflow.py`，以及对应 local/manual YAML overlay。
- 需要同步的文档与 specs：`docs/project_surface_inventory.md`、`docs/agent_navigation.md`、README 或相关 current docs、`openspec/specs/*` 中对旧入口和本地脚本面的声明。
- 验证重点：`openspec validate prune-stale-experiment-surfaces --strict`、架构边界测试、相关 sweep/BeamBench/配置加载 focused tests。
