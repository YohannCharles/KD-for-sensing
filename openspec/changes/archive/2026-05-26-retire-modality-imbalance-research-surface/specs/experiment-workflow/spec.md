## MODIFIED Requirements

### Requirement: Raymobtime s008 实验矩阵与分析 workflow
项目 MUST 提供 Raymobtime s008 推荐实验矩阵，用于运行和比较单模态、多模态、sensing-only、sensing+ray 和 task-aware gated 配置。项目 MUST 不再要求提供 Raymobtime s008 模态失衡分析 CLI、失衡诊断报告或失衡判定 workflow。

#### Scenario: 单模态与多模态矩阵
- **WHEN** 用户查看 Raymobtime s008 推荐配置或分析说明
- **THEN** 系统 MUST 覆盖 `coord` only、`image` only、`lidar` only、`ray` only，以及至少 `coord+image`、`coord+lidar`、`coord+ray`、`image+lidar`、`coord+image+lidar` 和 `coord+image+lidar+ray`
- **AND** 每组 MUST 能选择 simple concat 或 task-aware gated 模型

#### Scenario: sensing-only 单任务主矩阵
- **WHEN** 用户需要 Raymobtime s008 sensing-only 单任务主实验
- **THEN** 推荐运行矩阵 MUST 覆盖 `coord`、`image`、`lidar` 和 `coord+image+lidar` 四组输入条件
- **AND** 每组 MUST 分别运行 `current_beam_selection`、`current_los_classification` 和 `current_link_quality`
- **AND** 该矩阵共 12 个训练 run，包含 `ray` 的 sensing+ray run MUST 单独标注为补充实验

#### Scenario: 普通评估产物可比较
- **WHEN** Raymobtime s008 训练或评估完成
- **THEN** 系统 MUST 继续输出 objective 对应的 `metrics.json` 或 `test_report.json`
- **AND** 系统 MUST 不要求额外生成模态失衡 analysis CSV、drop modality delta 或 diagnosis report

## REMOVED Requirements

### Requirement: 项目健康检查中的 Phase 1.5 与互补分析覆盖
**Reason**: Phase 1.5 和互补分析已退役，健康检查不应继续要求这些测试。
**Migration**: 健康检查保留架构导入边界、console script help 和当前仍保留的核心诊断逻辑。

#### Scenario: 快速回归命令不再覆盖退役研究工具
- **WHEN** 开发者运行项目快速回归命令
- **THEN** 系统不再要求覆盖 Phase 1.5 pending gate 或互补分析核心测试
