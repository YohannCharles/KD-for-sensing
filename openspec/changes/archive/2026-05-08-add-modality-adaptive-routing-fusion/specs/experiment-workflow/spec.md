## MODIFIED Requirements

### Requirement: Teacher-prior CRAF validation subsets
验证流程 MUST 支持对支持 force modality mask 的 fusion 模型运行显式模态组合评估。该能力 MUST 只在模型支持 force modality mask 且配置启用时运行，并 MUST 支持从 teacher prior 或等价配置中解析 top-prior、single-best-prior 和 low-prior 模态集合。既有 CRAF 配置 MUST 继续可用，MARF 配置 MUST 使用同一验证入口。

#### Scenario: 运行 prior-driven strong-only 和 weak-only 验证
- **WHEN** 配置启用 `evaluation.modality_subsets` 且 teacher prior 可用
- **THEN** 验证流程 MUST 使用 force modality mask 分别评估 strong-only 和 weak-only 组合
- **AND** strong-only MUST 对应当前 prior 最高的 top-k 可用模态
- **AND** weak-only MUST 对应当前 prior 最低的一组可用模态
- **AND** strong-only 和 weak-only 的实际模态列表 MUST 记录到验证输出或运行日志

#### Scenario: all subset 与官方验证一致
- **WHEN** 配置请求 `all` subset
- **THEN** `all` subset MUST 使用全部启用模态执行与官方 validation 等价的 forward
- **AND** `val/subset/all/top1` MUST 与官方 `accuracy/val` 在同一 checkpoint 和 dataloader 上一致

#### Scenario: 支持 MARF subset 名称
- **WHEN** 配置请求 `top_prior`、`single_best_prior`、`random_with_top_prior` 或 low-prior subset
- **THEN** 验证流程 MUST 按 prior 和配置参数解析对应 force mask
- **AND** 验证结果 MUST 包含每个成功评估 subset 的 Top-1、ATop-3、ATop-5、ADBA 和 loss

#### Scenario: 非 opt-in 模型跳过模态组合验证
- **WHEN** 模型不支持 `supports_force_modality_mask`
- **THEN** 验证流程 MUST 跳过模态组合评估
- **AND** 默认验证指标 MUST 仍正常产出
