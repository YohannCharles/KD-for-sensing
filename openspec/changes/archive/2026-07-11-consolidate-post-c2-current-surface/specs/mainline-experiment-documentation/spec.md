## MODIFIED Requirements

### Requirement: 主线模型目录
项目 MUST 维护当前主线模型目录，用于集中说明 final C2 / U-MaskBeamJEPA、必要 baseline/control、MMW/CSI supporting workflow 和当前证据 owner 的研究问题、配置入口、数据口径、指标口径、运行状态与结果引用。目录 MUST 位于 `docs/mainline_model_catalog.md` 或等价 current 文档中，并 MUST 不把 retired、historical、supporting-only 或 mock-only workflow 描述为当前推荐入口。

#### Scenario: 主线模型目录覆盖当前支持面
- **WHEN** 开发者阅读主线模型目录
- **THEN** 文档 MUST 覆盖 final C2 / U-MaskBeamJEPA、U-Mask eval matrix、仍用于对照的 AMR/AMBER、MMW GPS v2、physics-informed MMW、CSI hardening 和 current paper evidence owner
- **AND** 每行 MUST 标明 config/入口、数据集/场景、split/target、metric profile、运行状态和 caveat

#### Scenario: 退役路线不得进入 current 主线表
- **WHEN** 主线模型目录提到 Image+GPS JEPA query-pool、Vision-Position、BeamBench、BEV-Fusion、KD、Hist、geometry prior 或旧 Scene31 workflow
- **THEN** 对应行 MUST 标记为 retired、historical 或 compatibility context
- **AND** 文档 MUST 不提供这些路线的 current 推荐训练命令

### Requirement: 主实验证据收敛记录
主线实验文档 MUST 在 final C2 / U-MaskBeamJEPA 进入证据收敛阶段时记录 final checklist、缺失 evidence、claim status 和下一步最小动作。文档 MUST 区分继续补证据与新增方法搜索，并在主方法冻结时说明冻结边界。

#### Scenario: Scene31-34 evidence 更新
- **WHEN** Scene31-34 final summary、paper tables 或 claim status 发生变化
- **THEN** mainline history、model catalog、experiment protocols 或 result claims registry 中的对应 current fact MUST 同步更新
- **AND** 真实 metrics、figures、logs 和 checkpoint MUST 继续留在 ignored output root

#### Scenario: Temporal evidence 更新
- **WHEN** H5/P1 temporal matrix 或 final C2 missing-modality evidence 从 smoke 转为可比较结果
- **THEN** 文档 MUST 更新 claim status、config/manifest provenance 和 caveat
- **AND** synthetic smoke 结果 MUST 继续标记为 mock/smoke

### Requirement: 主线文档必须反映 post-C2 边界
主线文档 MUST 将当前默认研究主线描述为 final C2 / U-MaskBeamJEPA 缺失模态波束预测，并明确 MMW/CSI 为保留的 current supporting dataset workflow。非主线历史复现、一次性诊断、research dashboard/preview 和已删除 CLI MUST 不再出现在 current recommended workflow 表中。

#### Scenario: current mainline 表述
- **WHEN** 开发者阅读 README、current research brief、mainline model catalog 或 experiment matrix
- **THEN** 文档 MUST 将 final C2 / U-MaskBeamJEPA、U-Mask eval matrix、claim/evidence gate 和保留 MMW/CSI workflow 描述为当前重点
- **AND** 文档 MUST 不把 Image+GPS JEPA、BeamBench、BEV-Fusion 2604、Vision-Position、geometry prior 或旧 RBMA/KD sweep 描述为当前默认主线

#### Scenario: MMW 保留状态清楚
- **WHEN** 文档列出 MMW/CSI 相关入口
- **THEN** 文档 MUST 标明保留原因、数据集用途、入口命令、输出边界和 focused validation
- **AND** 文档 MUST 不把 MMW/CSI 列为 post-C2 删除候选

## REMOVED Requirements

### Requirement: 项目描述反映当前主线
**Reason**: 该要求仍把已退役 Image+GPS JEPA、Vision-Position、BEV-Fusion 和旧诊断描述为 current，与 post-C2 requirement 冲突。
**Migration**: 使用更新后的“主线模型目录”和“主线文档必须反映 post-C2 边界”。

#### Scenario: 高层描述使用 post-C2 口径
- **WHEN** README 或项目 metadata 描述当前项目
- **THEN** 文档 MUST 使用更新后的 final C2 + MMW/CSI 口径
- **AND** 不再引用本旧 requirement

### Requirement: 当前推荐 workflow 聚焦少样本跨场景主线
**Reason**: 旧要求将多条已退役路线列为推荐 workflow，语义已被 post-C2 主线替代。
**Migration**: Current quickstart 只列保留的十个 CLI 和 protected workflow。

#### Scenario: Quickstart 不恢复旧 workflow
- **WHEN** 开发者阅读 quickstart
- **THEN** 文档 MUST 指向 final C2、U-Mask、MMW/CSI 和保留核心入口
- **AND** 旧 workflow 只能作为 historical context

### Requirement: 健康检查反映保留入口
**Reason**: 该要求仍强制 JEPA visual/GPS shortcut；健康检查现由 project-health-guardrails 的 post-C2 规则统一定义。
**Migration**: 使用更新后的 protected surface、CLI 和 stale-reference focused checks。

#### Scenario: Focused validation 使用 current surface
- **WHEN** 开发者运行 quick checks
- **THEN** 检查 MUST 不要求已退役 JEPA visual/GPS shortcut 或 surface doctor
- **AND** 保留入口仍必须通过验证

### Requirement: 当前支持面收敛到 Image+GPS JEPA query-pool
**Reason**: Image+GPS JEPA query-pool、Vision-Position、BeamBench 和 visual analysis 已在 post-C2 lifecycle 中退役。
**Migration**: 当前主线使用 final C2 / U-MaskBeamJEPA；MMW JEPA 仅保留已被 current config 消费的 mean pooling/pretraining 组件。

#### Scenario: 旧 query-pool 主线不再要求
- **WHEN** 开发者检查 current configs、CLI 或 model catalog
- **THEN** 项目 MUST 不要求 GPSQueryPool、fair_gps_biased、Vision-Position 或 JEPA visual CLI 存在
- **AND** 旧结果可从 archive/git 查询

### Requirement: Research dashboard 汇总 paper readiness
**Reason**: Research dashboard/HTML 产品面退役，paper readiness 由人工 claim registry 与 paper export gate 管理。
**Migration**: 使用 `docs/result_claims_registry.md`、`kd-sensing-runs` 和 `kd-sensing-paper-export`。

#### Scenario: Paper readiness 不依赖 dashboard
- **WHEN** 维护者判断 claim 是否可导出
- **THEN** 必须读取正式 claim/protocol 和 paper export gate
- **AND** 不要求 dashboard report 存在

### Requirement: RBMA ablation documentation
**Reason**: 旧独立 RBMA/prototype-KD ablation config matrix 与 runbook 已退役，继续维护推荐四配置和运行顺序会与 final C2 post-mainline 文档冲突；U-Mask 内嵌 RBMA/prototype/teacher branch 仍由其 current owner保护。
**Migration**: 历史实验计划与 caveat 从 archive/git 查询；current missing-pattern口径由 U-Mask/Scene31-34 owners记录。

#### Scenario: RBMA current 文档行退出
- **WHEN** mainline catalog、experiment matrix 和 claim registry 被检查
- **THEN** 它们 MUST 不把旧独立 RBMA configs 或运行顺序描述为 current/pending workflow
- **AND** 历史提及 MUST 标记 retired/historical

### Requirement: Harvested claim draft governance
**Reason**: Research claim harvester、candidate ledger 和 dashboard 产品面整体退役。
**Migration**: 人工维护 `docs/result_claims_registry.md`，并使用 run index、formal protocol 与 paper export gate。

#### Scenario: Claim docs 不依赖 harvester
- **WHEN** 维护者更新 claim status 或 paper evidence
- **THEN** current docs MUST 不要求 harvested candidate、dashboard summary 或 ledger record
- **AND** formal claim MUST 继续具备人工审阅 provenance
