## ADDED Requirements

### Requirement: U-MaskBeamJEPA 必须提供 opt-in PGCD block surface
U-MaskBeamJEPA MUST 在 `pgcd` 显式启用时复用 `[B,T,M,D]` latent 和同一 `BeamPrototypeBank`，输出 `[B,N,64]` block evidence、`[B,N]` quality/reliability/prior/fusion tensors 和 `[B,T,M]` availability。配置未声明 PGCD 时 MUST 不实例化 PGCD 参数，并保持默认 forward 与 state dict 行为不变。

#### Scenario: 默认 T2 forward
- **WHEN** canonical T2 未声明 `model.primary.pgcd`
- **THEN** model MUST 使用现有 current Router 路径
- **AND** 不得计算 PGCD clean/corrupted 双 view

#### Scenario: PGCD corrupted forward
- **WHEN** active PGCD config 提供 corrupted 四模态与 availability
- **THEN** fused logits MUST 可由共享 block evidence 与 PGCD fusion weights 重构
- **AND** missing block MUST 不参与 feature、evidence 或 logits 融合

### Requirement: PGCD quality reroute 必须只依赖部署时状态
模型 MUST 提供从 cached corrupted block features/evidence/availability 重新计算 PGCD weights 的接口，用于 D0-D3 替换。该接口 MUST 不接受 clean tensors、label、severity、corruption type 或 weather。

#### Scenario: 替换动态 reliability
- **WHEN** evaluator 提供 dynamic、train-fit global mean 或 reliability=1 replacement
- **THEN** reroute MUST 在同一 block evidence 上只改变 fusion weights
- **AND** 输出权重 MUST 满足 availability 归一化契约
