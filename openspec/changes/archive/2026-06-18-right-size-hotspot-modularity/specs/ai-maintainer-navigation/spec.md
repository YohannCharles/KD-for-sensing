## ADDED Requirements

### Requirement: 导航按右尺寸化决策处理热点
AI 维护导航文档 SHALL 指导 agent 和维护者在处理热点模块、长函数、长类、facade 或 helper 边界时使用右尺寸化决策矩阵。导航 MUST 明确拆分、合并、监控、接受当前尺寸和预算调整都是可能的有效动作，且 MUST 禁止把热点预算机械解释为“所有大文件都要拆”。

#### Scenario: agent 修改热点前先判断动作类型
- **WHEN** AI agent 准备修改已登记热点、接近预算的 workflow、dataset、diagnostic module 或 facade
- **THEN** 导航文档 MUST 要求先读取 `docs/maintainer_context_index.yaml` 中的 hotspot metadata
- **AND** agent MUST 判断本次变更属于拆分、合并/收敛、监控、接受当前尺寸、预算调整或源码窄修复中的哪一种

#### Scenario: facade 与业务模块使用不同判断
- **WHEN** agent 遇到公开 CLI/import facade 和真实业务 workflow 同时接近预算
- **THEN** 导航文档 MUST 要求 facade 继续按硬预算和防回流规则处理
- **AND** 业务 workflow MUST 按 rationale、headroom、validation commands 和调用边界判断是否拆分或保持线性流程

#### Scenario: 低价值抽象优先合并
- **WHEN** agent 发现单调用点包装类、只为减少行数的小 helper、重复 utils 聚合或无公开兼容价值的 facade
- **THEN** 导航文档 MUST 要求优先考虑合并回清晰 owner 或改为私有局部 helper
- **AND** agent MUST 不通过新增兼容包装层、旧入口或绕过 `src/kd_sensing` 包结构来完成合并

#### Scenario: 输出方案包含验证和风险
- **WHEN** agent 提出或执行热点右尺寸化变更
- **THEN** 方案或最终说明 MUST 写明选择拆分、合并、监控或接受当前尺寸的原因
- **AND** 方案 MUST 列出对应 focused validation commands，并说明不会读取真实 `dataset/` 或写入 ignored runtime artifacts

### Requirement: 导航支持高风险修复 campaign
AI 维护导航文档 SHALL 指导 agent 在用户明确接受高风险时使用 remediation wave 计划，而不是把多个热点重构混成单次不可定位的大改。导航 MUST 要求 agent 先确认 active OpenSpec change、读取维护上下文索引、列出 wave 顺序和每个 wave 的验证命令，再开始源码实施。

#### Scenario: 高风险请求转为 wave 计划
- **WHEN** 用户要求完整修复热点架构且明确可以接受高风险
- **THEN** agent MUST 将方案拆成多个 remediation waves
- **AND** 每个 wave MUST 标明目标文件、计划动作、保留或改变的 public surface、focused tests 和回滚/停止条件

#### Scenario: 当前打开文件不被孤立处理
- **WHEN** IDE 当前打开 `data_factory.py`、`sequences.py`、loss 或 model 文件
- **THEN** agent MUST 把这些文件放回维护上下文索引和 wave 计划判断
- **AND** agent MUST 不因为文件当前打开就默认拆分，也不得因为文件较小就忽略测试和 owner 边界

#### Scenario: 小模块作为 keep-and-test 样板
- **WHEN** agent 审核到小而内聚的 loss、model 或 helper 模块
- **THEN** 导航 MUST 允许选择 keep-and-test
- **AND** agent MUST 说明为什么不拆，以及需要补充或保留哪些 focused tests
