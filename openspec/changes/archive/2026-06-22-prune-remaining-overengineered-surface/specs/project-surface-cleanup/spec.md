## ADDED Requirements

### Requirement: Ponytail 审计候选必须有删除证据
项目 MUST 在删除或合并低价值源码、配置、脚本、测试或文档入口前记录最小证据：当前调用方、公开入口风险、替代 owner、是否被 registry/CLI/current docs/OpenSpec 消费、验证命令和回滚方式。没有证据的候选 MUST 保留或降为后续单独 change。

#### Scenario: 删除候选具备证据
- **WHEN** 开发者准备删除 ponytail 审计列出的候选项
- **THEN** change artifact 或实现说明 MUST 记录该候选不属于当前 package CLI、registry、canonical config、README/docs current 入口、OpenSpec current requirement 或必要 focused test 输入
- **AND** 记录 MUST 指向替代 owner、替代 recipe、普通 unknown-name 行为或说明无需替代

#### Scenario: 保留候选具备理由
- **WHEN** 某个候选因 public API、外部脚本风险、manifest 安全边界或当前 workflow 消费而保留
- **THEN** inventory、任务说明或最终实现说明 MUST 记录保留理由
- **AND** 项目 MUST 不为保留该候选新增兼容 wrapper、二级聚合层或重复治理表

### Requirement: 源码瘦身不得清理本地产物
本 change 的源码、测试、配置和文档删减 MUST 与本地产物清理分离。实现 MUST 不删除、移动或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或 TensorBoard 产物。

#### Scenario: 源码改动保护本地产物
- **WHEN** 本 change 删除或合并源码、测试、配置、脚本或文档
- **THEN** git diff MUST 不包含 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 中的新删除或修改
- **AND** 若用户另行要求清理这些路径，流程 MUST 使用 runtime cleanup manifest 或单独显式确认

### Requirement: 退役保活测试必须删除或迁移
只用于证明退役类、旧 alias、旧 facade 或已删除 helper 仍可直接导入/forward 的测试 MUST 删除或改写为当前 owner、registry unknown-name、canonical config 或 CLI 行为测试。

#### Scenario: 退役模型 direct-forward 测试不再保活
- **WHEN** 已从 registry 退役的整模型类不再属于当前公开 API
- **THEN** 测试 MUST 不再直接实例化该类来证明其 forward 仍可用
- **AND** 相关覆盖 MUST 迁到当前 `modular_sequence`、feature extractor、registry unknown-name 或 config load 行为
