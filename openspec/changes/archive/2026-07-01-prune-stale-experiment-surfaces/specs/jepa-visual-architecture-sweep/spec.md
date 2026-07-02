## ADDED Requirements

### Requirement: 当前 architecture sweep 入口优先
JEPA visual architecture sweep MUST 以 `jepa_visual_architecture_sweep` owner、`configs/diagnostics/jepa_visual_architecture_sweep_manifest.yaml` 和 `configs/fusion/experiments/jepa_image_gps/architecture_sweep_{smoke,lowmem,strict}.yaml` 作为当前推荐 sweep surface。旧 CNN/hybrid full sweep MAY 仅作为历史兼容 reader 保留，不得继续作为默认训练 runner。

#### Scenario: 模型摘要读取当前 manifest
- **WHEN** 用户通过模型架构摘要入口读取 sweep manifest
- **THEN** 系统 MUST 支持当前 `jepa_visual_architecture_sweep` manifest schema
- **AND** 系统 MUST 不要求旧 full sweep runner 存在

#### Scenario: 旧 full sweep 只读兼容
- **WHEN** 仍需读取 `cnn_hybrid_jepa_visual_prior_sweep` 历史 manifest
- **THEN** 系统 MAY 保留只读 manifest expansion 或 summary reader
- **AND** 该兼容路径 MUST 不生成训练 job、不清理 output root、不调度 GPU 任务

## REMOVED Requirements

### Requirement: CNN/hybrid visual-prior full sweep matrix
**Reason**: 该 full sweep 矩阵是旧的大规模实验生成 surface，当前推荐 architecture sweep 已收敛到更小的 manifest 和 focused 配置矩阵。继续要求完整 full mode 会保留低价值 2.7k 行 runner。
**Migration**: 使用 `jepa_visual_architecture_sweep` manifest 和 focused architecture sweep 配置；历史 full manifest 仅可通过只读兼容 reader 消费。

#### Scenario: full mode 不再是当前要求
- **WHEN** 开发者检查当前 JEPA visual architecture sweep 契约
- **THEN** 系统 MUST 不再要求 full mode 展开旧 `existing_controls`、`teacher_guided_stabilization` 和 `seed_confirm` 全量矩阵
- **AND** 当前候选覆盖 MUST 由当前 architecture sweep manifest 定义

### Requirement: Stage-aware job generation
**Reason**: 旧 full sweep 的 stage-aware job graph、Stage 1/downstream/teacher/re-evaluation 调度是执行 runner 逻辑，不再属于当前推荐 surface。
**Migration**: 对仍需运行的候选使用当前 `kd-sensing-train` / `kd-sensing-evaluate` 命令和 focused manifest；需要恢复大规模调度时另起 OpenSpec change。

#### Scenario: 旧 job graph 不再生成
- **WHEN** 用户加载当前 architecture sweep manifest
- **THEN** 系统 MUST 不要求生成旧 full sweep 的 Stage 1/downstream/teacher job graph
- **AND** 当前 manifest MAY 记录人类可执行命令或 focused command manifest

### Requirement: Full sweep runner safety and parallelism
**Reason**: 固定 GPU 0-3、最多 8 进程并行、resume、cleanup 的旧 runner 属于本地大规模实验编排，不应作为长期 current requirement。
**Migration**: 本地批量运行使用显式脚本、平台任务系统或新的专用 change；源码默认只保留当前 focused sweep 能力。

#### Scenario: 旧 runner 不作为 current CLI
- **WHEN** 用户查看当前 JEPA visual architecture sweep quickstart
- **THEN** 文档 MUST 不要求运行旧 `cnn_hybrid_jepa_visual_prior_sweep` runner
- **AND** 清理旧输出仍 MUST 通过 runtime cleanup manifest 或用户显式路径完成

### Requirement: Visual-prior summary, Pareto, and claim gate
**Reason**: 旧 full sweep summary/Pareto/seed aggregation 与旧全量矩阵绑定。当前 claim gate 和 summary 应使用轻量 architecture sweep 的统一字段。
**Migration**: 使用 `jepa_visual_architecture_sweep` 的 `strict_comparability_gate`、`summary_row_from_result` 和 `write_sweep_summary`。

#### Scenario: 当前 summary 使用轻量 schema
- **WHEN** architecture sweep 生成 summary
- **THEN** 系统 MUST 使用当前 architecture sweep summary schema
- **AND** 系统 MUST 不要求旧 full sweep 的 full table、family best、checkpoint-selection comparison 或 seed confirm 聚合存在
