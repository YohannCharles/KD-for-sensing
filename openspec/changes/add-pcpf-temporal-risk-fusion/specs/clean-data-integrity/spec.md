## ADDED Requirements

### Requirement: PCPF-T 可拟合风险状态只能来自绑定协议的 train role

PCPF-T 的风险分量 mean/std、静态能力先验、`mean_train_risk`、温度、解析融合参数和 checkpoint selection MUST 只使用所绑定 `mmw_id_stratified_block_v1` seed manifest 的 train role 与只读 validation role。所有可拟合统计 MUST 仅遍历 manifest 声明的 train windows，validation/test MUST 不更新模型、统计、阈值、prototype、memory bank 或 gate；默认运行 MUST 记录 `test_evaluated=false`。

#### Scenario: 准备 Stage 2 或 Stage 3

- **WHEN** trainer 调用 PCPF-T 的 stage preparation
- **THEN** preparation MUST 只接收 train dataset 和 train temporal-missing transform
- **AND** 产物 MUST 记录 protocol、split role、遍历范围和 train-only 状态

### Requirement: 历史 development evaluation 必须显式降级声明

PCPF-T MAY 对 validation 做只读诊断，但配置和全部开发报告 MUST 固定记录 `claim_ineligible: true`。test 只能由独立显式最终评估读取；confirmation、trainval、merged split 或 test-driven gate/融合/模型选择 MUST 被拒绝。

#### Scenario: 评估 validation 或显式 test

- **WHEN** resolved config 运行开发 validation 或显式最终 test
- **THEN** evaluator MUST 保持只读并输出 `claim_ineligible: true`
- **AND** 未显式授权的 test 请求 MUST 在 dataset 创建前失败

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
