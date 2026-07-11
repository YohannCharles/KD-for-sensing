## MODIFIED Requirements

### Requirement: 配置驱动实验
项目 MUST 提供配置文件驱动的训练、评估和预处理入口。配置 MUST 覆盖数据路径、模态、`model.primary`、current supervised/adaptation/JEPA/CSI/U-Mask/MMW 目标、训练参数和输出边界；当前 workflow MUST 不要求 GPS-query、JEPA visual/shortcut、旧 Scene31 BTAPA/night-grid/next-round 或 retired KD 路线。

#### Scenario: 使用配置启动 current 训练评估
- **WHEN** 用户通过 retained CLI 传入 current single-modality、fusion、U-Mask、JEPA pretraining、MMW 或 CSI 配置
- **THEN** 系统 MUST 构建配置声明的数据、model、loss、optimizer/evaluation owner
- **AND** 未启用模态、retired teacher/distiller 或 deleted diagnostic MUST 不成为依赖

#### Scenario: 使用 current supporting workflow
- **WHEN** 用户运行 JEPA pretraining/mean reuse、MMW GPS/physics、CSI hardening 或 U-Mask eval matrix
- **THEN** 系统 MUST 使用对应 current package owner 和 config contract
- **AND** 系统 MUST 不恢复 GPS-query pooling、JEPA visual/shortcut、legacy KD/Hist/BGAM/viewer 或旧 Scene31 workflow

## REMOVED Requirements

### Requirement: BTAPA tau1 seed 与 es20 配置族
**Reason**: 旧 Scene31 BTAPA sweep 已退出 current provenance。
**Migration**: 历史配置/结果从 archive/git 查询；final C2/U-Mask 和 Scene31-34 为 current owners。
#### Scenario: BTAPA config family 退出
- **WHEN** current configs 被枚举
- **THEN** 项目 MUST 不要求该 seed/es20 family

### Requirement: 固定 GPU shell launcher 已收敛为直接命令
**Reason**: Requirement 仍要求旧 BTAPA/night-grid configs、fresh-eval helper 和 apples-to-apples module；整个 workflow 已退役。
**Migration**: Current 实验使用 retained package CLI 或 protected manifest runner。
#### Scenario: 旧直接命令不再推荐
- **WHEN** docs 被检查
- **THEN** 它们 MUST 不推荐旧 BTAPA/night-grid/apples-to-apples paths

### Requirement: proto vs BTAPA seed mean±std 分析
**Reason**: 专属 analysis script 与旧 sweep 同时退出。
**Migration**: Current Scene31-34 statistics owner管理正式多 seed evidence。
#### Scenario: BTAPA analysis 退出
- **WHEN** current scripts 被枚举
- **THEN** 专属 analyzer MUST 不存在

### Requirement: Scene31 night grid config generation
**Reason**: 58/64-run night grid 已退役且没有 current claim provenance。
**Migration**: 无；未来实验矩阵需新 change。
#### Scenario: Night grid generator 退出
- **WHEN** current generators 被枚举
- **THEN** 项目 MUST 不要求 night-grid generator/manifest

### Requirement: night grid generated configs are local artifacts
**Reason**: 对应 generator/workflow 整体删除，不再需要 artifact contract。
**Migration**: 一般 generated artifact boundary 仍适用于 current generators。
#### Scenario: Night-grid local artifacts 不再要求
- **WHEN** current workflow 运行
- **THEN** 它 MUST 不依赖 night-grid generated YAML

### Requirement: night grid fresh eval
**Reason**: 旧 eval script、checkpoint resolver facade 和 missing-pattern wrapper 均退出。
**Migration**: 使用 retained evaluate/U-Mask matrix/Scene31-34 owners。
#### Scenario: Night-grid eval 退出
- **WHEN** current evaluation entrypoints 被枚举
- **THEN** `eval_night_grid.py` MUST 不作为 current path

### Requirement: night grid analysis
**Reason**: 专属 ranking/report 没有 current consumer。
**Migration**: 历史结论保留在 mainline history/archive。
#### Scenario: Night-grid report 退出
- **WHEN** current reports 被枚举
- **THEN** 项目 MUST 不要求其 top-candidate artifacts

### Requirement: summary 兼容 night grid
**Reason**: Night-grid status compatibility 只维持已退役 manifest。
**Migration**: Current summary 只解析其明确 current artifact schema。
#### Scenario: Night-grid status schema 退出
- **WHEN** current summary 运行
- **THEN** 它 MUST 不要求 old night-grid manifest fields

### Requirement: Scene31 next-round local follow-up workflow
**Reason**: Scene31 next-round package 与 shared summary owner 整体退役。
**Migration**: 使用 protected Scene31-34 final workflow。
#### Scenario: Next-round workflow 退出
- **WHEN** current configs/scripts/docs 被枚举
- **THEN** 它们 MUST 不要求 Scene31 next-round launcher、config lookup 或 summary
