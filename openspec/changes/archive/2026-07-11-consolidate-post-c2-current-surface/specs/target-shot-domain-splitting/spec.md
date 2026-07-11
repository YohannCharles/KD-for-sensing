## ADDED Requirements

### Requirement: Target-shot split 作为 MMW supporting owner 保留
`target-shot-domain-splitting` MUST 分类为 supporting capability，因为 `kd_sensing.data.mmw.protocol` 直接复用其 deterministic split、leakage guard、artifact write/load 和 metadata contract。项目 MUST 保留这些 helper，但 MUST 不恢复已退役 standalone target-shot CLI、console script 或独立 quickstart workflow。

#### Scenario: MMW protocol 复用 target-shot helper
- **WHEN** MMW cross-scene protocol 构建 target labeled/unlabeled/test split artifact
- **THEN** protocol MUST 继续复用 `target_shot_splits.py` 的 canonical helper
- **AND** split determinism、sample overlap、target-test leakage 和 artifact fingerprint 行为 MUST 保持

#### Scenario: Supporting helper 不扩大为 public workflow
- **WHEN** pyproject、CLI help、README 和 current workflow 被枚举
- **THEN** 项目 MUST 不声明 standalone target-shot 命令或推荐入口
- **AND** lifecycle inventory MUST 将 capability 标记为 `supporting` 而不是 `retired-tombstone`
