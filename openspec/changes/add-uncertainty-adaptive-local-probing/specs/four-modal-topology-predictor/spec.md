## ADDED Requirements

### Requirement: predictor 必须是原生四模态单阶段模型

系统 MUST 仅以 `image、radar、gps、lidar` 五帧历史和 availability/temporal mask 预测一个 64 类未来 beam posterior。模型 MUST 只包含四个独立 encoder、唯一共享 Temporal Transformer、唯一 Beam Prototype Bank 与确定性 posterior statistics；MUST NOT 包含 CSI encoder、evidential/risk head、样本级模态权重 Router、static prior、temperature/tau 或多 stage 状态。预注册 static reliability 诊断/修复 MAY 增加恰好四个样本无关的 trainable fusion logits。

#### Scenario: 构建模型

- **WHEN** registry 构建 `four_modal_topology_predictor`
- **THEN** state dict MUST 不含 CSI、risk、concentration、temperature、tau 或样本级 fusion 参数
- **AND** mean control MUST 不含 fusion-weight 参数，static reliability 分支 MUST 只增加 `fusion_logits[4]`
- **AND** 旧 model id 或旧字段 MUST 失败，不得兼容加载

### Requirement: fused posterior 必须使用受控的 availability-masked probability fusion

每个可用模态 MUST 通过同一个 prototype bank 产生 64 类 probability。mean control 的 `fused_probability` MUST 为可用模态 probability 的 arithmetic mean。static reliability 分支 MUST 只使用四个全局 trainable logits，并在每个 availability mask 内 softmax 归一化；bounded 修复 MUST 先对 logits 使用无系数 `tanh`，旧无界诊断分支 MUST 保持原参数化语义。两者都 MUST NOT 读取样本 feature、posterior uncertainty、weather、domain、GT 或 RF measurement。缺失模态 MUST 对融合无贡献，任一样本至少一个模态可用，Single mask MUST 精确等于对应单模态 probability。

#### Scenario: 任意非空 missing mask

- **WHEN** forward 接收 `[B,4]` availability 或 `[B,5,4]` temporal mask
- **THEN** `unimodal_probabilities` MUST 为 `[B,4,64]` 且 `fused_probability` MUST 为 `[B,64]`
- **AND** posterior statistics MUST 只由该 fused probability 无状态派生

#### Scenario: static reliability 初始化与训练

- **WHEN** 构建 static reliability 修复模型
- **THEN** 四个 fusion logits MUST 以零初始化，使初始 forward 与 arithmetic mean 完全一致
- **AND** logits MUST 仅由 train loss 更新并随 checkpoint 保存；validation/test MUST 不更新或覆盖它们
- **AND** mean control MUST 不创建 fusion-weight parameter

#### Scenario: bounded static reliability 防止权重塌缩

- **WHEN** 构建 `bounded_static_reliability` 并对 Full mask forward
- **THEN** effective logits MUST 为 `tanh(fusion_logits)`，每个值严格位于 `[-1,1]`
- **AND** 任一 Full-mask 模态权重 MUST 不超过 `exp(2)/(exp(2)+3)`
- **AND** 初始化与 mean 完全一致，Single mask 仍 MUST 精确产生 one-hot fusion weight

### Requirement: topology supervision 必须是唯一可切换创新因素

单阶段 loss MUST 包含 fused hard CE、availability-normalized unimodal hard CE，并在 topology-on 时增加环形 soft CE 与 fused/modality prototype alignment。topology-off MUST 只将这些 topology 项置零；模型结构、训练预算、数据、mask schedule 和 checkpoint selection MUST 相同。

#### Scenario: matched topology on/off

- **WHEN** 两个配置仅切换 topology supervision
- **THEN** trainable parameter names/count、encoder、prototype bank、optimizer 与 epoch budget MUST 相同
- **AND** 两者均 MUST fresh-start，不得从旧五模态或多 stage checkpoint 初始化

### Requirement: masked-feature 主线必须使用唯一联合拓扑原型loss

`masked_feature_mlp` 的正式创新点1 MUST 将融合特征和全部可用单模态特征查询同一Prototype Bank所得的环形soft CE等权平均，并且只乘一次 `joint_topology_weight`。Hard-CE control MUST 将该权重置零。该主线 MUST 将旧 `unimodal_soft_weight`、`lambda_proto` 与 `lambda_modality_proto` 全部置零，禁止把相同soft目标重复计权。

#### Scenario: 联合拓扑loss计算

- **WHEN** 一个样本具有任意非空availability mask
- **THEN** 系统 MUST 计算一个fused topology soft CE，以及按该样本可用模态数归一化的unimodal topology soft CE
- **AND** `joint_topology = 0.5 * (fused_topology + unimodal_topology)`
- **AND** 总loss MUST 只增加 `joint_topology_weight * joint_topology`

#### Scenario: seed1 matched go/no-go

- **WHEN** 比较Hard-CE与Joint-Topology seed1
- **THEN** 两者 MUST 使用相同masked-feature模型参数、protocol、whole-modality schedule、optimizer、40 epoch与validation-best选择
- **AND** 除 `joint_topology_weight={0,0.1}` 外不得改变MLP宽度、sigma或其他loss权重

### Requirement: static reliability 修复必须使用 matched fresh 对照

bounded static reliability 与 arithmetic mean 的最终比较 MUST 使用相同 MMW protocol、whole-modality missing schedule、optimizer、epoch、seed、topology loss 与 validation-best selection。比较 MUST 覆盖 topology off/on 三 seed并报告四个静态权重、15-mask Direct/Posterior Top-3/TBCP-3；不得只选择 Full mask 或有利 seed。无界分支只作为预注册塌缩诊断，不得与 bounded 结果混合。

#### Scenario: 修复有效性验证

- **WHEN** static reliability fresh runs 完成 validation replay
- **THEN** 系统 MUST 同时报告 Full、missing-radar、Drop-1、Drop-2、Single、Radar-only 与 All-15 结果
- **AND** 结果 MUST 标记 `claim_ineligible=true`、`outer_test_accessed=false`

### Requirement: 标准 feature fusion 必须在共享原型空间联合单模态与融合特征

`masked_feature_mlp` MUST 在分类前将不可用模态特征置零，并将四个模态特征与 availability mask 拼接后通过固定两层MLP产生一个64维融合特征。该分支 MUST NOT 输出或使用显式四模态gate/reliability权重，也不得读取 posterior uncertainty、weather、domain、GT、CSI、历史beam或RF measurement。

模型 MUST 只有一个64-beam Prototype Bank。四个单模态特征与融合特征 MUST 查询同一个Bank：单模态由 availability-normalized hard/neighbor-soft loss监督，融合特征由 fused hard 与 topology prototype loss监督。系统 MUST NOT 为融合特征复制第二套prototype，亦不得把单模态同形态prototype soft loss重复计权来伪造联合聚类。

#### Scenario: 任意非空mask下联合原型forward

- **WHEN** `masked_feature_mlp` 接收 `[B,4]` availability 与 `[B,4,64]` 模态特征
- **THEN** 不可用特征 MUST 在进入MLP前严格为零，MLP输入 MUST 显式包含同一availability
- **AND** `unimodal_logits` MUST 为共享Bank查询四个单模态特征所得 `[B,4,64]`
- **AND** `fused_probability` MUST 为融合特征查询同一Bank所得归一化 `[B,64]`
- **AND** state dict MUST 只有一个 `prototype_bank.prototypes`

#### Scenario: topology off/on matched训练

- **WHEN** feature-fusion topology off/on 对同一batch backward
- **THEN** 两者模型参数名与数量 MUST 完全一致，均包含相同MLP和唯一Prototype Bank
- **AND** topology开关 MUST 只改变neighbor-soft/prototype loss贡献，不得改变融合架构或信息权限

### Requirement: evaluation 必须原生覆盖 15 个 mask

validation evaluator MUST 对四模态全部 15 个非空 availability mask使用同一 validation-best checkpoint和同一 validation identity/order。evidence metadata MUST 声明恰好四个 modalities 和 15 patterns；五模态、CSI row、31 patterns 或旧 checkpoint MUST 拒绝。

#### Scenario: 生成 probing prior evidence

- **WHEN** evaluator 收集完整 validation evidence
- **THEN** 每个 pattern MUST 恰好包含完整 validation 样本并保存归一化 `fused_probability[64]`
- **AND** 不得读取 channel、beam power、GT 之外的 metric-only radio information 来生成 sensing posterior

### Requirement: DeepSense6G secondary transfer 必须使用线性邻接

DeepSense6G secondary panel MAY 复用四模态 masked-feature Prototype-only predictor，但 MUST 使用 `linear_index_v1` label-index topology，MUST NOT 连接beam 0与63，也 MUST NOT 携带MMW ULA descriptor/audit。该迁移 MUST 使用固定过滤后的Scene31–34 pooled train/test、40 epoch和last checkpoint；它 MUST NOT 被表述为ULA物理拓扑复现。

#### Scenario: 构建DeepSense6G Prototype-only transfer

- **WHEN** `four_modal_topology_predictor` 的dataset为DeepSense6G
- **THEN** model与loss topology MUST同时为 `linear_index_v1` 且physical audit fields MUST为空
- **AND** fusion MUST为 `masked_feature_mlp`，`unimodal_soft_weight=0`、`lambda_proto=0.1`、`lambda_modality_proto=0`
- **AND** checkpoint selection MUST为last，不得虚构validation-best

#### Scenario: DeepSense6G topology误绑定

- **WHEN** DeepSense6G配置声明cyclic/permuted/ULA topology、MMW protocol/audit或best-validation选择
- **THEN** config MUST失败关闭
