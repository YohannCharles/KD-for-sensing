## ADDED Requirements

### Requirement: 维护索引记录本次支持面收敛结果
维护上下文索引 SHALL 记录本次删减后的 entrypoint、hotspot、merge-candidate、dependency 和 remediation wave 状态。索引 MUST 将 package console scripts 作为当前入口事实，MUST 不继续登记已删除的 Python thin alias 为 current entrypoint。

#### Scenario: entrypoint 索引不保留 thin alias
- **WHEN** `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py` 或 BeamBench thin alias 从源码删除
- **THEN** 维护索引 MUST 删除或重新分类对应 script entry
- **AND** package CLI 索引 MUST 与 `pyproject.toml` 的 `[project.scripts]` 保持双向一致

#### Scenario: 删除和合并候选有行动元数据
- **WHEN** 索引记录 `communication_state_features`、LiDAR pillar 原型、dataset runtime adapter 框架或重复 `OutputRegistry` 的收敛状态
- **THEN** entry MUST 标明 planned action、public surface policy、validation commands 和 rollback note
- **AND** public surface policy MUST 能区分 `remove-internal-only`、`merge-into-owner` 和 `keep-public-import`

#### Scenario: CSI hardening matrix 分类更新
- **WHEN** CSI hardening 配置矩阵从重复实体 YAML 收敛为 base+overlay 或 recipe
- **THEN** 索引或 inventory MUST 记录 base config、overlay/recipe 位置、当前配置 ID 范围和验证命令
- **AND** 架构边界测试 MUST 不再要求每个矩阵 ID 都对应一份完整实体 YAML

#### Scenario: dev dependency audit 可追踪
- **WHEN** dev extra 删除未使用依赖
- **THEN** 维护索引或 inventory MUST 记录该删除不影响 runtime dependencies
- **AND** 若后续重新引入同类依赖，必须在对应 change 中说明当前使用点和验证命令
