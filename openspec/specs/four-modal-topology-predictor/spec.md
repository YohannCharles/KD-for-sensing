# Four-Modal Topology Predictor Specification

## Purpose

定义原生四模态、单阶段的 topology-prototype sensing posterior predictor，并为 sensing-guided finite probing 提供冻结先验。

## Requirements

### Requirement: 模型必须严格使用四个 sensing 模态

系统 MUST 以 canonical `image、radar、gps、lidar` 五帧历史预测一个 64 类未来 beam。模型 MUST 使用四个独立 encoder、唯一共享 Temporal Transformer 与唯一 Beam Prototype Bank；MUST NOT 构造 CSI encoder、evidential/risk head、learned modality fusion、temperature/tau 或 training stage 参数。

#### Scenario: 构建或恢复 predictor

- **WHEN** registry 构建 `four_modal_topology_predictor`
- **THEN** state dict MUST 不含 CSI、risk、concentration、temperature、tau 或 fusion-weight 参数
- **AND** 旧 model id、旧配置字段和旧 checkpoint MUST 失败，不得自动迁移

### Requirement: temporal missing 必须正确屏蔽

encoder 输出 MUST 为 `[B,5,64]`，stack 为 `[B,5,4,64]`，并送入唯一共享 Temporal Transformer。false temporal cell MUST 不作为 attention key/value；完整缺失模态的 frame/CLS feature、probability 和贡献 MUST 显式为零。每个样本 MUST 至少有一个有效模态。

#### Scenario: 任意合法 temporal mask

- **WHEN** `modality_temporal_mask` 为 `[B,5,4]`
- **THEN** forward MUST 输出 `[B,4,64]` unimodal probability 与 `[B,64]` fused probability
- **AND** unavailable 模态 MUST 不影响 fused posterior

### Requirement: fused posterior 必须是无参数 masked mean

每个可用模态 MUST 查询同一个 Beam Prototype Bank 产生 probability。`fused_probability` MUST 是可用模态 probability 的 arithmetic mean；Single mask MUST 精确等于对应单模态 probability。MAP、circular mean/resultant/variance、beam variance/spread、entropy 与稳定 Top-L MUST 从 fused probability 在 FP32 中无状态派生。

#### Scenario: availability 改变

- **WHEN** 同一样本分别以 Full 与任一非空 missing mask forward
- **THEN** 模型 MUST 只重新执行相同四模态路径和 masked mean
- **AND** 不得调用 Router、risk、static prior 或 sample-wise learned fusion

### Requirement: topology supervision 必须是唯一创新开关

单阶段 loss MUST 包含 fused hard CE 和按可用模态数归一化的 unimodal hard CE。topology-on MUST 额外使用 ULA-DFT phase-cycle 邻近 soft CE 与 fused/modality prototype alignment；topology-off MUST 只将这些 topology 项置零。两个配置 MUST 使用相同模型参数、数据、mask schedule、optimizer、epoch budget和 validation checkpoint selection 并 fresh-start。

#### Scenario: matched on/off backward

- **WHEN** topology-on/off 对同一 synthetic batch backward
- **THEN** 两者 trainable parameter names/count MUST 相同且梯度有限
- **AND** topology-off loss MUST 不包含 soft/prototype contribution

### Requirement: evaluation 必须原生生成十五个 mask

validation evaluator MUST 对四模态全部 15 个非空 availability mask 使用同一 validation-best checkpoint 和相同完整 validation identity/order。evidence modalities MUST 恰好为 canonical 四模态，patterns MUST 恰好为 15；五模态、CSI row、31 patterns 或不完整样本 MUST 失败。

#### Scenario: 生成 probing evidence

- **WHEN** evaluator 收集完整 validation matrix
- **THEN** 每个 pattern MUST 保存 sample id、label、normalized fused probability 和 prediction
- **AND** test MUST 默认封存，报告 MUST 记录 `claim_ineligible=true` 与 `outer_test_accessed=false`
