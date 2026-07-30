## ADDED Requirements

### Requirement: PCPF-T 可拟合风险状态只能来自绑定协议的 train role

PCPF-T 的风险分量 mean/std、静态能力先验、`mean_train_risk`、温度、解析融合参数和 checkpoint selection MUST 只使用绑定 protocol 的 train role 与只读 validation role。sparse-CSI 正式路线 MUST 绑定 `mmw_trajectory_disjoint_v1` 的 37,510 个 train window 与 6,365 个 validation window；所有可拟合统计 MUST 仅遍历 train，validation MUST 不更新模型、统计、阈值或 gate，2,985 个 sealed test window MUST 不被构建或读取。

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

### Requirement: PCPF-T 输入必须保持历史 sensing-only

PCPF-T 默认模型、风险 target 和 stage preparation MUST 只消费 canonical image、radar、gps、lidar、temporal availability mask 与未来 beam label。只有配置显式声明 `use_sparse_csi=true` 时，模型 MAY 额外消费从同一样本五帧历史 `csi1..csi5` channel 引用按固定 2x2 选择确定性生成的 sparse CSI。当前/未来 CSI、未来 channel、path、beam power、历史 beam、天气、场景和 corruption metadata MUST 不得进入 forward、风险 target 或可拟合统计；天气与 domain MAY 仅作为评估分组元数据。

#### Scenario: 配置携带禁止字段

- **WHEN** PCPF-T model、loss 或 risk 配置声明任一禁止输入
- **THEN** 严格配置校验 MUST 在模型和 dataset 创建前失败

#### Scenario: 构建 sparse CSI batch

- **WHEN** PCPF-T 配置显式启用历史 sparse CSI
- **THEN** dataset MUST 验证五个 channel 引用分别匹配历史 frame id 且最后历史帧早于 target
- **AND** 生成与编码 MUST 不注入 AWGN、dropout、corruption、当前/未来 CSI 或虚构 SNR
- **AND** metadata MUST 记录 split role、selection/codebook identity 与 `snr_available=false`
