## ADDED Requirements

### Requirement: probing topology likelihood 只能由 train radio ground truth 拟合

系统 MAY 为 sensing-guided finite RF probing 拟合一个独立 topology likelihood artifact，但其输入 MUST 仅为绑定 `mmw_id_stratified_block_v1` train role 的官方未来 64-beam power 和对应 argmax beam label。artifact MUST 只描述 ULA-DFT phase-cycle 上按最优 beam 对齐的相对 log-gain 均值/协方差与 normalized-gain kernel；MUST NOT 进入 PCPF-T forward、loss、optimizer、checkpoint selection、prototype、risk 或 sensing posterior。validation/test power、label 或指标 MUST NOT 更新、选择或校准该 artifact。

#### Scenario: 拟合 probing likelihood artifact

- **WHEN** evaluator 准备 TBCP-7 train-only calibration
- **THEN** fitter MUST 只遍历 protocol audit 绑定的完整 train sample identity，并验证 sample count/hash、label 与 power argmax、topology 和 source hashes
- **AND** artifact MUST 记录 `fit_split=train`、protocol/topology identity、有效样本数与内容 fingerprint

#### Scenario: 尝试从 validation 或 test 拟合

- **WHEN** artifact source role 不是 train，或其 sample identity/count 与 train audit 不一致
- **THEN** 系统 MUST 在读取目标 split 的完整 power 前失败关闭
- **AND** 不得通过 confirmation、trainval、validation replay 或 test sensitivity 更新 artifact

#### Scenario: 使用 probing artifact 推理

- **WHEN** TBCP-7 在 validation 或显式最终评估中加载 artifact
- **THEN** artifact MUST 保持只读且独立于模型 state dict
- **AND** loader MUST 重新核对当前 train power 内容 fingerprint、manifest/source CSV hash 与 topology descriptor/audit identity，任一漂移 MUST 失败关闭
- **AND** candidate policy MUST 只接收 sensing prior、已请求 measurement 和该 train-only topology statistics，不得接收完整 evaluation power vector
