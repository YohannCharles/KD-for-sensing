## ADDED Requirements

### Requirement: 主线文档必须反映 post-C2 边界
主线文档 MUST 将当前默认研究主线描述为 final C2 / U-MaskBeamJEPA 缺失模态波束预测，并明确 MMW/CSI 作为保留的 future/current supporting dataset workflow。非主线历史复现、一次性诊断和删除候选 MUST 不再出现在 current recommended workflow 表中。

#### Scenario: current mainline 表述
- **WHEN** 开发者阅读 README、`docs/current_research_brief.md`、`docs/mainline_model_catalog.md` 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 将 final C2 / U-MaskBeamJEPA 缺失模态主线、缺失模态 eval matrix、claim/evidence gate 和保留 MMW workflow 描述为当前重点
- **AND** 文档 MUST 不把已删除或待删除的 Image+GPS JEPA、BeamBench、BEV-Fusion 2604、Vision-Position 或旧 RBMA/KD sweep 描述为当前默认主线

#### Scenario: MMW 保留状态清楚
- **WHEN** 文档列出 MMW/CSI 相关入口
- **THEN** 文档 MUST 标明其保留原因、数据集用途、入口命令、输出边界和 focused validation
- **AND** 文档 MUST 不把 MMW 支线列为 post-C2 删除候选

### Requirement: 删除候选文档必须迁移或降级
被 post-C2 清理删除的历史报告、runbook、README 段落或实验矩阵行 MUST 被迁移为 concise historical note，或从 current docs 删除。保留的历史说明 MUST 不提供当前推荐运行命令。

#### Scenario: 历史报告删除前迁移结论
- **WHEN** implementation 删除历史 Markdown、results log 或一次性报告
- **THEN** 仍有价值的结论 MUST 迁移到 `docs/mainline_experiment_history.md`、`docs/result_claims_registry.md`、inventory historical note 或等价 current 文档
- **AND** 迁移后的说明 MUST 标记为 historical、retired、blocked 或 caveat，而不是 promoted claim

#### Scenario: 当前 docs 不引用删除 runbook
- **WHEN** historical `scripts/` runbook 或 package CLI 被删除
- **THEN** README、experiment matrix、protocol table 和 current OpenSpec specs MUST 不再推荐该命令
- **AND** 若需要复跑历史实验，文档 MUST 指向 preserved generator/base config、final C2 入口、MMW 入口或明确 historical note

### Requirement: Claim provenance 保护主线 YAML/manifest
主线 claim/protocol 文档 MUST 继续指向真实存在且受保护的 YAML/manifest，或指向等价 generator/base config。删除配置不得使 pending 或 reviewed claim 失去 provenance。

#### Scenario: provenance 输入被保护
- **WHEN** `docs/result_claims_registry.md`、`docs/experiment_protocols.md` 或 `docs/mainline_model_catalog.md` 引用 YAML/manifest 作为主线 evidence
- **THEN** implementation MUST 保留该 YAML/manifest 或先更新 provenance 到等价输入
- **AND** claim status MUST 不因删除历史文件而自动升级

#### Scenario: 不确定主线用途时暂缓删除
- **WHEN** implementation 无法确认某个 YAML/manifest 是否会被用户主线使用
- **THEN** 该文件 MUST 标记为 `pending-confirmation` 或 `protected-until-next-audit`
- **AND** 本 change MUST 不删除该文件
