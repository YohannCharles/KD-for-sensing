## ADDED Requirements

### Requirement: 架构边界测试验证结构化事实而非 prose mirror
项目健康护栏 MUST 将长期稳定事实放入机器可读索引、inventory lifecycle、OpenSpec requirements、pyproject 或 AST/path/import 扫描中验证。架构边界测试 MUST 不逐字镜像 README、docs 或 OpenSpec 的自然语言段落，除非该文本本身是退役 token、公开入口名、路径、命令或需要静态拒绝的 lifecycle wording。

#### Scenario: 文档自然语言改写不触发结构测试失败
- **WHEN** README 或 docs 在不改变入口、路径、lifecycle、命令、配置引用或退役语义的情况下改写说明文字
- **THEN** 架构边界测试 MUST 不因固定短语缺失而失败
- **AND** 测试 MUST 继续验证机器可读索引、路径和 OpenSpec lifecycle 是否一致

#### Scenario: 当前入口事实仍被验证
- **WHEN** README、docs、OpenSpec 或维护索引声明当前 CLI、配置路径、dataset type、模型注册名或诊断入口
- **THEN** 架构边界测试 MUST 验证对应路径、pyproject entry point、索引 entry 或源码 owner 存在
- **AND** stale 当前入口引用 MUST 失败

#### Scenario: 退役 wording guard 保留
- **WHEN** current docs 或 current specs 将已退役路线写成 quickstart、active mainline、默认 workflow 或长期入口
- **THEN** 健康护栏 MUST 继续失败
- **AND** 失败信息 MUST 指向加入退役限定、更新 lifecycle 或删除推荐入口

#### Scenario: 护栏检查无运行副作用
- **WHEN** 开发者运行架构边界测试或文档健康检查
- **THEN** 检查 MUST 只读取已跟踪源码、配置、文档、OpenSpec artifact、pyproject 和测试文件
- **AND** 检查 MUST 不读取真实 `dataset/`、`outputs/`、`logs/`、checkpoint、cache 或 TensorBoard event
