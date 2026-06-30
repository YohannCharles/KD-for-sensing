## ADDED Requirements

### Requirement: 主线文档记录 AMBER full pending 状态
主线模型目录、实验协议表和结果 claim 账本 MUST 记录 AMBER full architecture reproduction 的本地 pending 状态、配置入口、输出边界、指标口径和 caveat。文档 MUST 区分 AMBER-lite、AMBER full local architecture reproduction 和任何未来 official AMBER reproduction。

#### Scenario: 主线目录包含 AMBER full 条目
- **WHEN** AMBER full 配置和 focused tests 落地
- **THEN** `docs/mainline_model_catalog.md` 或等价 current 文档 MUST 记录其 model line、config、入口命令、数据集/场景、metric profile、run status 和 caveat
- **AND** 该条目 MUST 标记为 local architecture reproduction 或 pending，直到真实严格可比结果存在

#### Scenario: claim 账本不写入未验证数值
- **WHEN** AMBER full 只有 synthetic tests、dry-run 或未完成训练
- **THEN** `docs/result_claims_registry.md` MUST NOT 填入真实性能数值
- **AND** 它 MUST 只记录 pending/unverified 状态、输出路径边界和升级条件
