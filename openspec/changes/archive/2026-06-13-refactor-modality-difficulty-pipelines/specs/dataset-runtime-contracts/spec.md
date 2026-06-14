## ADDED Requirements

### Requirement: Runtime metadata 记录 difficulty profile
Dataset/dataloader runtime metadata MUST 记录当前 run 实际启用的 difficulty profiles。metadata MUST 包含 profile id、stage、split、operator types、affected modalities、resolved severities、seed、digest、fallback 和 warnings summary。未启用 difficulty 时，metadata MUST 明确记录 clean/default 状态或省略为兼容旧行为。

#### Scenario: train dataloader 记录 difficulty profile
- **WHEN** 训练配置为 train split 启用 GPS mild async profile
- **THEN** dataloader 或 run metadata MUST 记录该 profile 的 resolved stage/split、operator、seed 和 digest
- **AND** validation/test split 若未启用 profile，MUST 不被标记为同一扰动条件

#### Scenario: 未启用 difficulty 保持 clean
- **WHEN** 配置没有声明 difficulty profile
- **THEN** dataset/dataloader 构建 MUST 保持现有 clean 输入行为
- **AND** runtime metadata MUST 不要求新增非空 difficulty 字段才能被下游消费

### Requirement: Difficulty transform 不改变 target contract
Runtime dataset、dataloader 或 batch transform 应用 difficulty profile 时，MUST 保持 target provider 输出的主 label、辅助 target、valid mask、sample id、split 和 dataset family metadata 不变。GPS delay、stride、dropout 或 image degradation MUST 只影响输入模态字段及其输入相关 mask/metadata。

#### Scenario: GPS delay 不移动 target
- **WHEN** batch 应用 GPS delay、low-rate 或 async difficulty
- **THEN** `target_beam`、`target_beam_distribution`、`beam_power`、auxiliary target 和 sample id MUST 与 clean batch 一致
- **AND** runtime metadata MUST 记录该 difficulty 作用于 GPS 输入而非 target schema

#### Scenario: image degradation 不改变 input profiles
- **WHEN** batch 应用 image fog/rain、night、occlusion 或 motion blur
- **THEN** resolved image input profile MUST 仍是原配置对应 profile
- **AND** metadata MUST 将 degradation 记录为 difficulty condition，而不是新的 dataset profile 或新模态

### Requirement: Difficulty 作用阶段边界
系统 MUST 支持按 train、validation、test、evaluation 和 benchmark stage/split 选择 difficulty profile。stage/split 选择 MUST 在 dataloader metadata 和 run metadata 中可审计，MUST 防止训练 profile 隐式泄漏到 evaluation-only benchmark，或 evaluation sweep 隐式改变训练数据。

#### Scenario: evaluation profile 不影响训练 dataloader
- **WHEN** 配置只声明 evaluation difficulty sweep
- **THEN** train dataloader MUST 使用 clean 输入
- **AND** evaluation 或 benchmark pass MUST 按 sweep profile 应用 difficulty

#### Scenario: train profile 不影响 benchmark manifest
- **WHEN** 训练 run 使用 mild async profile，但 benchmark manifest 未引用该 profile
- **THEN** benchmark MUST 不自动继承训练 difficulty profile
- **AND** benchmark metadata MUST 只记录 manifest 显式声明的 difficulty conditions
