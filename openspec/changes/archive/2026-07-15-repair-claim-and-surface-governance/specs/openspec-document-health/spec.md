## ADDED Requirements

### Requirement: Lifecycle inventory 必须与实际 current surface 双向一致
OpenSpec 和文档健康检查 MUST 对 current specs、root/current documents 和 agent context 的 actual 与 declared lifecycle 集合执行双向比较。检查 MUST 分别报告 missing、extra、duplicate 和非法 lifecycle，MUST NOT 只验证 inventory 是 actual 的子集。

#### Scenario: 新 current spec 未登记
- **WHEN** `openspec/specs/<capability>/spec.md` 存在但 lifecycle inventory 没有该 capability
- **THEN** architecture/document health test MUST 失败
- **AND** diagnostics MUST 报告 missing capability 名称

#### Scenario: Inventory 保留不存在 spec
- **WHEN** lifecycle inventory 声明 current/supporting capability，但对应 current spec 不存在
- **THEN** test MUST 失败并报告 extra capability

#### Scenario: Root/current document 未分类
- **WHEN** 仓库根或声明的长期 docs 中出现未分类 Markdown/PDF-facing source document
- **THEN** document health test MUST 失败
- **AND** inventory MUST 要求 current、historical、generated/local 或 delete 分类及 owner

#### Scenario: Agent context 引用退役入口
- **WHEN** current agent context 或 AGENTS 推荐未在 `project.scripts`/current script lifecycle 中存在的命令
- **THEN** test MUST 失败并报告文档、命令和推荐语境

## MODIFIED Requirements

### Requirement: Root 文档支持面分类
项目 MUST 对仓库根目录、current 长期文档和 agent context 进行生命周期分类。当前 README MUST 保持安装、快速上手和主 workflow；长期需求与架构约束 MUST 留在 OpenSpec；研究、环境和复现记录 MUST 标明 current、historical、adapter 或 scoped-context 边界。

#### Scenario: Root Markdown 与 agent context 有完整生命周期
- **WHEN** 开发者运行文档健康检查
- **THEN** inventory MUST 精确覆盖 on-disk root Markdown 与 `docs/agent_context/*.md`
- **AND** 每个记录 MUST 包含 lifecycle、owner 和 purpose

#### Scenario: 退役 root 报告只保留迁移账目
- **WHEN** 旧复现报告的代码 owner 和入口已经删除
- **THEN** inventory MUST 记录其 delete/migrated 处置和替代历史账本
- **AND** 不存在的 root 报告 MUST 不再被 current README、agent context 或推荐命令引用

#### Scenario: 文档不推荐退役入口
- **WHEN** README、AGENTS 或长期 docs 描述当前可运行 workflow
- **THEN** 文档 MUST 不把已退役 KD/HiST/Top8/residual/BeamBench/旧 JEPA diagnostic 路线描述为当前推荐入口
- **AND** 如需保留历史背景，文档 MUST 明确标记为历史、退役或防回流记录
