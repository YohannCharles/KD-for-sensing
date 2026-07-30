## ADDED Requirements

### Requirement: PCPF-T 可拟合风险状态只能来自 inner train

PCPF-T 的风险分量 mean/std、静态能力先验、`mean_train_risk`、温度、解析融合参数和 checkpoint selection MUST 只使用绑定 protocol 的 `inner_train` 与只读 `inner_validation`。所有可拟合统计 MUST 仅遍历 `inner_train`；validation MUST 不更新模型、统计、阈值或 gate。

#### Scenario: 准备 Stage 2 或 Stage 3

- **WHEN** trainer 调用 PCPF-T 的 stage preparation
- **THEN** preparation MUST 只接收 train dataset 和 train temporal-missing transform
- **AND** 产物 MUST 记录 protocol、split role、遍历范围和 train-only 状态

### Requirement: 历史 development evaluation 必须显式降级声明

PCPF-T MAY 对已在历史开发中使用的 development split 做只读诊断，但配置和全部报告 MUST 固定记录 `claim_ineligible: true`。outer test、confirmation、trainval 或 merged split MUST 不得被构建、读取或用于调参、gate、融合和模型选择。

#### Scenario: 评估 historical development split

- **WHEN** resolved config 将 split 标记为 historical development
- **THEN** evaluator MUST 保持只读并输出 `claim_ineligible: true`
- **AND** 任何 outer-test 请求 MUST 在 dataset 创建前失败

### Requirement: PCPF-T 输入必须保持 sensing-only

PCPF-T 模型、风险 target 和 stage preparation MUST 只消费 canonical image、radar、gps、lidar、temporal availability mask 与未来 beam label。channel、CSI、path、beam power、历史 beam、天气、场景和 corruption metadata MUST 不得进入 forward、风险 target 或可拟合统计；天气与 domain MAY 仅作为评估分组元数据。

#### Scenario: 配置携带禁止字段

- **WHEN** PCPF-T model、loss 或 risk 配置声明任一禁止输入
- **THEN** 严格配置校验 MUST 在模型和 dataset 创建前失败
