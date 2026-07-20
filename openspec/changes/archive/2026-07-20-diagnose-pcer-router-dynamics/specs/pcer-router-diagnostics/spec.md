## ADDED Requirements

### Requirement: 诊断必须冻结并复用历史实验身份
系统 MUST 从已完成 PCER 快速验证目录加载每个实验的最佳 checkpoint 及其 resolved config，并在固定 split、样本身份和 mask 身份下执行只读推理。系统 MUST NOT 更新模型参数、覆盖历史产物或从测试集拟合替代权重。

#### Scenario: 运行权重替换评测
- **WHEN** 诊断 A1 或 A3 的 D0-D4 融合模式
- **THEN** 所有模式 MUST 共享同一次模型 evidence、同一批标签和同一 availability mask
- **AND** D1/D2 统计 MUST 来自 validation split

### Requirement: 权重替换必须保持实际融合语义
系统 MUST 为 D0 使用 checkpoint 的动态权重，为 D1 使用全局平均 router logits，为 D2 使用 mask 条件平均 router logits，为 D3 使用 A0 静态融合语义，为 D4 使用所有可用融合单元的均匀权重。所有模式 MUST 在 softmax 前屏蔽不可用单元并重新归一化。

#### Scenario: A3 发生不均匀时间块缺失
- **WHEN** A3 在某模态内只有部分时间块可用
- **THEN** D3 MUST 先在可用模态间均分质量，再在每个模态的可用时间块间均分
- **AND** D4 MUST 对所有可用时间块直接均分

### Requirement: counterfactual target 必须可独立验证
系统 MUST 通过 synthetic tests 验证 `loss_without_i - loss_all` 的贡献符号、`KL(target || prediction)` 的参数方向、缺失块双侧屏蔽以及 time-major 展平索引，并在真实样本上统计 target 分布、质量相关性和 router 对齐。

#### Scenario: 唯一正确块和唯一有害块
- **WHEN** synthetic evidence 分别包含唯一正确块或唯一有害块
- **THEN** 唯一正确块 MUST 获得最大 target 权重
- **AND** 唯一有害块 MUST 获得明显较低 target 权重

#### Scenario: 最高贡献块不可用
- **WHEN** availability mask 屏蔽最高贡献块
- **THEN** target 与 prediction 在该位置 MUST 同为零
- **AND** 其余可用位置权重和 MUST 为一

### Requirement: router 梯度审计不得改变 checkpoint
系统 MUST 对固定单 batch 仅执行一次 `L_route.backward()`，记录 router、prototype 和 backbone 的梯度范数、参数范数及加权损失量级，但 MUST NOT 调用 optimizer step 或保存 checkpoint。

#### Scenario: target 已 detach
- **WHEN** route loss 对 frozen diagnostic model 反向传播
- **THEN** router prediction 路径 MUST 接收有限非零梯度
- **AND** detached target/evidence 路径 MUST NOT 意外向 prototype 或 backbone 传播该 loss 梯度

### Requirement: S3 诊断必须保留具体缺失模态
系统 MUST 对 A0-A3 的每个整模态缺失场景分别报告预测指标，并对 A1/A3 输出剩余模态权重迁移。系统 MUST 对 A3 worst 子场景输出 beam 距离、混淆和标签/预测分布，不得只报告 S3 macro 或 worst 聚合值。

#### Scenario: 生成 S3 结论
- **WHEN** 四个整模态缺失场景评测完成
- **THEN** 诊断 MUST 明确 worst 缺失模态、邻近与远距离错误比例、权重集中情况
- **AND** MUST 比较逐块贡献之和与整模态 leave-one-out 贡献的非加性残差

### Requirement: 诊断产物必须可审计且保持本地
系统 MUST 在 `outputs/quick_pcer_diagnostics/` 写出命令、checkpoint、配置、样本数、CSV/JSON/Markdown 和最终结论，并将证据标记为单 seed development diagnosis。系统 MUST NOT 将本地输出提升为正式 claim evidence。

#### Scenario: 诊断完成
- **WHEN** 统一诊断入口成功退出
- **THEN** 用户要求的全部产物 MUST 存在且非空
- **AND** 最终总结 MUST 回答 A1、A3 target、S3 和 A-F 方向判断
