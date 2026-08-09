## ADDED Requirements

### Requirement: predictor 必须是原生四模态单阶段模型

系统 MUST 仅以 `image、radar、gps、lidar` 五帧历史和 availability/temporal mask 预测一个 64 类未来 beam posterior。模型 MUST 只包含四个独立 encoder、唯一共享 Temporal Transformer、唯一 Beam Prototype Bank 与确定性 posterior statistics；MUST NOT 包含 CSI encoder、evidential/risk head、模态权重 Router、static prior、temperature/tau 或多 stage 状态。

#### Scenario: 构建模型

- **WHEN** registry 构建 `four_modal_topology_predictor`
- **THEN** state dict MUST 不含 CSI、risk、concentration、temperature、tau 或 fusion-weight 参数
- **AND** 旧 model id 或旧字段 MUST 失败，不得兼容加载

### Requirement: fused posterior 必须为可用模态 probability 均值

每个可用模态 MUST 通过同一个 prototype bank 产生 64 类 probability；`fused_probability` MUST 为可用模态 probability 的 arithmetic mean。缺失模态 MUST 对均值无贡献，任一样本至少一个模态可用，Single mask MUST 精确等于对应单模态 probability。

#### Scenario: 任意非空 missing mask

- **WHEN** forward 接收 `[B,4]` availability 或 `[B,5,4]` temporal mask
- **THEN** `unimodal_probabilities` MUST 为 `[B,4,64]` 且 `fused_probability` MUST 为 `[B,64]`
- **AND** posterior statistics MUST 只由该 fused probability 无状态派生

### Requirement: topology supervision 必须是唯一可切换创新因素

单阶段 loss MUST 包含 fused hard CE、availability-normalized unimodal hard CE，并在 topology-on 时增加环形 soft CE 与 fused/modality prototype alignment。topology-off MUST 只将这些 topology 项置零；模型结构、训练预算、数据、mask schedule 和 checkpoint selection MUST 相同。

#### Scenario: matched topology on/off

- **WHEN** 两个配置仅切换 topology supervision
- **THEN** trainable parameter names/count、encoder、prototype bank、optimizer 与 epoch budget MUST 相同
- **AND** 两者均 MUST fresh-start，不得从旧五模态或多 stage checkpoint 初始化

### Requirement: evaluation 必须原生覆盖 15 个 mask

validation evaluator MUST 对四模态全部 15 个非空 availability mask使用同一 validation-best checkpoint和同一 validation identity/order。evidence metadata MUST 声明恰好四个 modalities 和 15 patterns；五模态、CSI row、31 patterns 或旧 checkpoint MUST 拒绝。

#### Scenario: 生成 probing prior evidence

- **WHEN** evaluator 收集完整 validation evidence
- **THEN** 每个 pattern MUST 恰好包含完整 validation 样本并保存归一化 `fused_probability[64]`
- **AND** 不得读取 channel、beam power、GT 之外的 metric-only radio information 来生成 sensing posterior
