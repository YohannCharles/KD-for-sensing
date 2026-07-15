## ADDED Requirements

### Requirement: Explicit MMW domain list dataset construction
数据运行时 MUST 支持通过 `data.dataset.domains` 显式声明多个 MMW condition/scenario domain。每个 domain MUST 作为独立 MMWDataset 构建后再组成 pooled dataset，且 MUST 保留自己的 data root、split CSV、condition、scenario 和 runtime metadata。

#### Scenario: 构建跨天气 pooled train dataset
- **WHEN** 配置包含 sunny、rainy、foggy 共 15 个有效 MMW domain
- **THEN** train dataset MUST 是 15 个 leaf MMWDataset 的组合
- **AND** 每个 leaf MUST 使用该 domain 声明的 train CSV 与 condition root
- **AND** runtime metadata MUST 输出 domain id、condition、scenario、split path 和 sample count

#### Scenario: domain 声明不完整
- **WHEN** 某个 domain 缺少 id、condition、scene、data root 或当前 split CSV
- **THEN** dataset construction MUST 在 DataLoader worker 启动前失败
- **AND** 错误 MUST 标出具体 domain 与缺失字段

### Requirement: Optional domain-balanced train sampler
当配置启用 `data.domain_balanced_sampling.enabled=true` 时，训练 DataLoader MUST 对 pooled domain 使用可复现的等 domain 权重 sampler；validation/test DataLoader MUST 保持确定性全量遍历，不得应用 replacement sampling。

#### Scenario: train sampler 可复现
- **WHEN** 两次运行使用相同 experiment seed、domain inventory 和 sample counts
- **THEN** domain-balanced sampler MUST 产生相同的抽样序列
- **AND** run metadata MUST 记录 sampler type、replacement、num samples 和 seed

