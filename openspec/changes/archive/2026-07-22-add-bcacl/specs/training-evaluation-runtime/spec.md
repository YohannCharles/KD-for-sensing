## ADDED Requirements

### Requirement: BCACL detached two-stage 使用显式 model-only initialization
runtime MUST 将 BCACL Phase 1 和 Phase 2 作为两个可独立恢复的运行。Phase 2 MUST 从 Phase 1 checkpoint 严格执行 model-only initialization，重置 optimizer、scheduler、epoch、RNG 与 sampler 轨迹，并记录 source SHA、role、schema 和 stage provenance。

#### Scenario: 启动 Phase 2
- **WHEN** Phase 1 `last.pth` 已完成且 Phase 2 launcher 校验 source identity
- **THEN** Phase 2 MUST 加载完整兼容模型权重并从 epoch 0 构建新的冻结 optimizer
- **AND** validation/test 不得参与 Phase 1 原型或质量状态形成

#### Scenario: 阶段身份不一致
- **WHEN** source checkpoint SHA、dataset、模型 key/shape 或 BCACL stage 不符合 Phase 2 请求
- **THEN** runtime MUST 在训练前失败

### Requirement: BCACL checkpoint 可恢复统计状态
checkpoint MUST 保存 BCACL 可训练参数、模态原型、初始化 mask、样本计数、质量矩阵和质量有效 mask；extension state MUST 保存影响后续确定性行为的配置与阶段身份。

#### Scenario: Phase 1 精确续跑
- **WHEN** 从 Phase 1 `last.pth` 恢复下一 epoch
- **THEN** 原型和质量状态 MUST 与保存前一致
- **AND** 后续教师选择 MUST 使用恢复状态而非 validation/test 重估

### Requirement: fixed-mask 评估提供 15-pattern 明细和分组汇总
评估 MUST 保留 full 加 14 个非空不完整组合的逐 pattern 结果，并输出 Full、Single/Double/Triple Macro 与 Worst、All-14 Macro 与 Worst。每个 pattern MUST 保留 Top-1、Top-3、Top-5、Within-3 和 MAE；已有通信效用字段在数据提供时 MUST 保留。

#### Scenario: 完整 15-pattern 评估
- **WHEN** fixed-mask evaluator 完成全部合法四模态非空组合
- **THEN** 输出 MUST 同时包含 15 条 pattern 明细和各分组 macro/worst
- **AND** 任一组合缺失时汇总 MUST 标记不完整而非伪造数值

### Requirement: BCACL 快速实验按晋级顺序执行
launcher MUST 先完成 smoke，再完成 U1/U2，再完成固定教师，最后才允许自动教师；任务 MUST 绑定 single seed、inner/development split、输出目录和 GPU identity，已有 GPU 任务不得被抢占。

#### Scenario: 固定教师失败
- **WHEN** 固定教师任务未成功产生有限损失、checkpoint 和完整 development fixed-mask 结果
- **THEN** launcher MUST 不启动自动教师任务
