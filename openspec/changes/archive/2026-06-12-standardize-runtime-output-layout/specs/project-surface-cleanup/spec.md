## MODIFIED Requirements

### Requirement: 输出目录用途分区
项目 MUST 为新的本地运行产物提供用途清晰的输出目录约定。训练 run、evaluation artifact、analysis artifact、cache、features、cleanup/organize manifest、scene/scenegroup best checkpoint 和 legacy archive MUST 采用可识别分区；当前支持 workflow MUST 不默认向语义不清的 `outputs/other/`、根级 `outputs/<run_name>/`、数字场景根 `outputs/31/` 或根级 `outputs/best_checkpoints/` 写入新产物。

Canonical 分区 MUST 至少包含：

- `outputs/cache/`: 可再生成 runtime cache。
- `outputs/cleanup_manifests/`: cleanup 或 organize dry-run/execution manifest。
- `outputs/analysis/`: 长期诊断、论文图、聚合分析和机器可读分析报告。
- `outputs/visual_analysis/`: JEPA visual analysis MAY 暂时保留为当前诊断出口；若迁移，MUST 迁入 `outputs/analysis/` 下的明确子目录。
- `outputs/evaluations/`: 评估集合和评估矩阵输出。
- `outputs/scene<id>/`: 单场景训练 run 和该 scene 的 best checkpoint registry。
- `outputs/scenegroup_<range-or-list>/`: 多场景训练 run 和该 scenegroup 的 best checkpoint registry。
- `outputs/archive/`: legacy root run、legacy `eval_*`、legacy numeric scene root 和需要保留但不作为当前入口的本地产物。

#### Scenario: 新训练输出写入可识别 scope
- **WHEN** 当前支持的训练 workflow 未显式覆盖输出目录
- **THEN** 默认输出 MUST 写入可识别的 scene 或 scenegroup 训练目录
- **AND** 运行目录 MUST 继续保存 `final_config.yaml`、`resolved_config.yaml`、metrics、checkpoint 和 runtime metadata
- **AND** 运行目录 MUST 不写入根级 `outputs/<run_name>/`、`outputs/31/<run_name>/` 或 `outputs/other/<run_name>/`

#### Scenario: 清理和整理 manifest 写入固定分区
- **WHEN** 用户运行 runtime cleanup 或 output organize dry-run
- **THEN** manifest MUST 写入 `outputs/cleanup_manifests/` 或用户显式指定的 manifest 路径
- **AND** manifest 路径 MUST 不与训练 run、evaluation、analysis、cache、features 或 registry checkpoint 混放

#### Scenario: legacy archive 不作为当前入口
- **WHEN** 本地存在迁移后的 `outputs/archive/` 产物
- **THEN** README、当前 docs、默认配置和 registry 解析 MUST 不把 archive 路径作为当前推荐入口
- **AND** 如需引用 archive 中的历史产物，文档 MUST 明确标记为历史记录或人工复核材料
