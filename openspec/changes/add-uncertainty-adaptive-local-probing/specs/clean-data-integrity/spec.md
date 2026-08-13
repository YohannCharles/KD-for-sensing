## ADDED Requirements

### Requirement: final MMW test panel必须先封存再读取

系统 MUST 在构建test dataset前完成四方法三seed完整性检查并写入包含checkpoint/config/protocol/topology/normalization/likelihood/policy SHA256的final-test seal。任何方法、seed或身份不完整 MUST 失败关闭；test只能用于一次性只读final evaluation，不得进入训练、checkpoint选择、calibration或超参数修改。

#### Scenario: final panel预检失败

- **WHEN** 12个冻结run任一缺失、digest漂移、不是validation-best或仍带`test_evaluated=true`
- **THEN** 系统 MUST 在读取test CSV、label、power前终止
- **AND** seal MUST 记录失败但不得生成部分test排名

### Requirement: DeepSense6G secondary split 必须显式过滤并固定末次评测

系统 MUST 只使用绑定manifest的Scene31–34官方train/test窗口。`future_beam1`非64维、非finite或负值的窗口 MUST由预生成过滤CSV和manifest显式排除，不得在dataset worker中静默跳过、插值或用零替换。由于不存在兼容五帧validation，所有方法 MUST固定40 epoch并选择last checkpoint；test只允许在训练方案冻结后执行一次最终只读评测。

#### Scenario: 使用过滤后的secondary protocol

- **WHEN** 生成DeepSense6G三模型实验配置
- **THEN** config MUST绑定protocol manifest、每scene train/test CSV SHA256、pooled count `13240/4090`与过滤规则
- **AND** 三种方法与全部seed MUST共享完全相同的split identity

#### Scenario: test参与选择

- **WHEN** 训练或调参流程尝试用test loss选择epoch、checkpoint、topology或超参数
- **THEN** 流程 MUST失败或结果 MUST判定为不合格

### Requirement: probing topology likelihood 只能由 train radio ground truth 拟合

系统 MAY 为 sensing-guided finite RF probing 拟合一个独立 topology likelihood artifact，但其输入 MUST 仅为绑定 `mmw_id_stratified_block_v1` train role 的官方未来 64-beam power 和对应 argmax beam label。artifact MUST 只描述 ULA-DFT phase-cycle 上按最优 beam 对齐的相对 log-gain 均值/协方差与 normalized-gain kernel；MUST NOT 进入 four-modal predictor forward、loss、optimizer、checkpoint selection、prototype 或 sensing posterior。validation/test power、label 或指标 MUST NOT 更新、选择或校准该 artifact。

#### Scenario: 拟合 probing likelihood artifact

- **WHEN** evaluator 准备 TBCP-3 train-only calibration
- **THEN** fitter MUST 只遍历 protocol audit 绑定的完整 train sample identity，并验证 sample count/hash、label 与 power argmax、topology 和 source hashes
- **AND** artifact MUST 记录 `fit_split=train`、protocol/topology identity、有效样本数与内容 fingerprint

#### Scenario: 尝试从 validation 或 test 拟合

- **WHEN** artifact source role 不是 train，或其 sample identity/count 与 train audit 不一致
- **THEN** 系统 MUST 在读取目标 split 的完整 power 前失败关闭
- **AND** 不得通过 confirmation、trainval、validation replay 或 test sensitivity 更新 artifact

#### Scenario: 使用 probing artifact 推理

- **WHEN** TBCP-3 在 validation 或显式最终评估中加载 artifact
- **THEN** artifact MUST 保持只读且独立于模型 state dict
- **AND** loader MUST 重新核对当前 train power 内容 fingerprint、manifest/source CSV hash 与 topology descriptor/audit identity，任一漂移 MUST 失败关闭
- **AND** candidate policy MUST 只接收 sensing prior、已请求 measurement 和该 train-only topology statistics，不得接收完整 evaluation power vector
