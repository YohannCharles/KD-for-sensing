## ADDED Requirements

### Requirement: 健康护栏验证维护上下文索引
项目健康护栏 SHALL 验证维护上下文索引存在、格式可读、必填 section 齐全，并与 AGENTS、AI 维护导航、project surface inventory、OpenSpec specs 和源码路径保持一致。检查 MUST 不读取真实数据、不启动训练、不写入本地产物。

#### Scenario: 索引缺失或不可解析
- **WHEN** 开发者运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **THEN** 测试 MUST 验证维护上下文索引存在且可解析
- **AND** 缺失、YAML 格式错误或必填 section 缺失时测试 MUST 失败并说明修复路径

#### Scenario: AGENTS 和 inventory 未引用索引
- **WHEN** 维护上下文索引已经存在
- **THEN** 健康护栏 MUST 验证 `AGENTS.md`、`docs/agent_navigation.md` 或 `docs/project_surface_inventory.md` 中有稳定引用或分类说明
- **AND** 缺少引用时测试 MUST 失败，避免索引成为无人读取的旁路文件

### Requirement: 健康护栏从索引读取治理 allowlist
项目健康护栏 SHALL 将长期治理 allowlist 和热点预算的数据来源迁移到维护上下文索引。测试 MAY 继续包含断言逻辑，但 MUST 不在测试文件中维护与索引重复的长期事实表。

#### Scenario: 脚本入口检查使用索引
- **WHEN** 架构边界测试检查 `scripts/`、`tools/analysis/` 或 shell orchestration 文件
- **THEN** 允许入口及其 lifecycle MUST 来自维护上下文索引
- **AND** 新增未登记入口 MUST 失败并提示更新索引和必要文档

#### Scenario: 配置、模型和 batch/runtime 检查使用索引
- **WHEN** 架构边界测试检查 root fusion config、整模型注册、batch/runtime 分支或 hotspot budget
- **THEN** 对应 allowlist 或 budget MUST 来自维护上下文索引
- **AND** 测试 MUST 继续拒绝未登记的新公开入口、未说明例外的新整模型注册和未登记热点扩张

### Requirement: 索引一致性检查不放宽退役路线护栏
项目健康护栏 SHALL 在读取维护上下文索引后继续拒绝 retired route 回流。索引中的 retired token、migration guard 和 forbidden-entry 分类 MUST 用于增强检查，而不是允许旧 KD、HiST/Hist、Top8 standalone、GPS residual、camera residual、Raymobtime s008、CRAF/MARF/G2D 或 Multimodal-NF 重新成为当前入口。

#### Scenario: retired route 被登记为当前入口
- **WHEN** 维护上下文索引、README、inventory、pyproject 或 configs 中把退役路线登记为 current quickstart、root config、console script 或长期 workflow
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 要求改为 retired/supporting/migration guard 语义或删除该入口

#### Scenario: migration guard 合法引用被允许
- **WHEN** 索引或 specs 只在 migration guard、历史说明、拒绝边界或 retired tombstone 中提到退役路线
- **THEN** 健康检查 MUST 允许该引用
- **AND** 检查 MUST 不把合法拒绝说明误判为入口回流
