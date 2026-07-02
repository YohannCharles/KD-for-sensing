## ADDED Requirements

### Requirement: Local/manual 实验面必须可收敛
本地 Scene31、RBMA missing-modality、strong encoder checkpoint 复用和 M2Beam 单模态训练 overlay MUST 被分类为 local/manual experiment surface。它们 MAY 保留为人工运行材料，但 MUST 有 owner、输出边界、删除触发条件和不升级为 package CLI 的说明。

#### Scenario: 固定 GPU queue shell 分类
- **WHEN** `scripts/run_next_v3_experiments.sh`、`scripts/run_rbma_strong_encoder_4gpu_queue.sh` 或 `scripts/run_m2beam_single_modal_scene31_queue.sh` 被保留
- **THEN** inventory MUST 将其分类为 local/manual shell orchestration
- **AND** 文档 MUST 说明输出仅允许进入 ignored `logs/` 和 `outputs/scene31/`

#### Scenario: 统一 runner 覆盖后删除 shell
- **WHEN** 一个 local/manual runner 或文档命令已经覆盖同等配置列表、dry-run、并发和 resume 需求
- **THEN** 固定 GPU shell MAY 被删除
- **AND** 删除 MUST 同步更新 inventory、docs 和架构边界测试

#### Scenario: overlay YAML 有删除触发条件
- **WHEN** `configs/scene31/`、`configs/fusion/experiments/rbma_missing_workflow_strong_encoders/` 或 `configs/fusion/experiments/m2beam_single_modal_scene31/` 中的 YAML 被保留
- **THEN** inventory 或 docs MUST 记录其 local/manual owner 和删除触发条件
- **AND** 这些 YAML MUST 不作为 root canonical fusion 入口

### Requirement: 本地实验结论沉淀后删除临时配置
当 local/manual 实验的关键结论、指标 provenance 和 caveat 已进入 result registry、experiment matrix 或报告文档时，对应临时 queue overlay 和脚本 MUST 收敛为删除或历史记录。

#### Scenario: 结果进入 registry
- **WHEN** Scene31/RBMA/M2Beam 本地实验的 promoted 或 pending claim 已写入 `docs/result_claims_registry.md`
- **THEN** 只服务该结论的临时脚本或 overlay MAY 被删除
- **AND** 保留的复跑路径 MUST 指向 package CLI、owner module 或明确的 local/manual runner

#### Scenario: checkpoint 占位不可升级 claim
- **WHEN** strong-encoder overlay 引用本地 `outputs/scene31/best_checkpoints/*.pth` 占位
- **THEN** 该配置 MUST 保持 local/manual 或 blocked/pending 状态
- **AND** 文档 MUST 不把缺 checkpoint 的路径声明为可复现 mainline claim
