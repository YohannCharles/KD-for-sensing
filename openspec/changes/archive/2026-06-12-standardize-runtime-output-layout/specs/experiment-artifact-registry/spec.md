## MODIFIED Requirements

### Requirement: 场景隔离的最佳 checkpoint registry
最佳 checkpoint registry MUST 按 DeepSense6G scene 或 scenegroup 隔离。默认 registry 目录 MUST 位于当前输出 scope 下，例如 `outputs/scene9/best_checkpoints/`、`outputs/scene31/best_checkpoints/`、`outputs/scene32/best_checkpoints/`、`outputs/scenegroup_s32_s34/best_checkpoints/` 和 `outputs/scenegroup_s31_s34/best_checkpoints/`。根级 `outputs/best_checkpoints/` MUST 只作为 legacy registry 输入由整理 manifest 审计，不得作为当前默认写入目标。

#### Scenario: Scenario 9 registry 写入 scene9
- **WHEN** 用户运行 Scenario 9 strong 训练并产生新的最高验证 Top-1 checkpoint
- **THEN** 系统 MUST 将归档 checkpoint 写入 `outputs/scene9/best_checkpoints/`
- **AND** metadata sidecar MUST 记录 `scene_id: 9` 和 `scene_slug: scene9`

#### Scenario: 默认 Scenario 31 registry 不复用其它场景
- **WHEN** 用户运行默认 Scenario 31 评估配置且未显式指定绝对 checkpoint
- **THEN** 系统 MUST 优先查找 `outputs/scene31/best_checkpoints/`
- **AND** 系统不得默认加载 `outputs/scene9/best_checkpoints/`、`outputs/scene32/best_checkpoints/` 或任一 scenegroup registry 中同 slug 的 checkpoint

#### Scenario: Scenario 32 registry 不复用 scene31
- **WHEN** 用户运行显式 Scenario 32 评估配置且未显式指定绝对 checkpoint
- **THEN** 系统 MUST 优先查找 `outputs/scene32/best_checkpoints/`
- **AND** 系统不得默认加载 `outputs/scene31/best_checkpoints/` 或任一 scenegroup registry 中同 slug 的 checkpoint

#### Scenario: 多场景 registry 写入 scenegroup
- **WHEN** 用户运行覆盖 scenes 32、33、34 的多场景训练并产生新的最高验证 Top-1 checkpoint
- **THEN** 系统 MUST 将归档 checkpoint 写入 `outputs/scenegroup_s32_s34/best_checkpoints/`
- **AND** metadata sidecar MUST 记录 scene scope 和参与的 train/validation/test scenes

#### Scenario: 多场景评估不回退到单场景 registry
- **WHEN** 用户运行多场景评估配置且未显式指定绝对 checkpoint
- **THEN** 系统 MUST 优先查找匹配 scenegroup 的 registry
- **AND** 系统不得默认加载 `outputs/scene31/best_checkpoints/` 中同 slug 的 checkpoint 作为替代

#### Scenario: legacy 根级 registry 进入整理复核
- **WHEN** 整理 manifest 扫描到 `outputs/best_checkpoints/`
- **THEN** manifest MUST 将其标记为 legacy registry
- **AND** 只有当 sidecar metadata 能唯一确定目标 scene 或 scenegroup 且目标无冲突时，manifest MAY 建议迁移到 canonical registry
- **AND** 否则 manifest MUST 标记为人工复核或 archive

#### Scenario: 显式绝对 checkpoint 仍最高优先级
- **WHEN** 用户通过绝对路径显式指定 teacher checkpoint 或评估权重
- **THEN** 系统 MUST 使用该显式路径
- **AND** scene 或 scenegroup registry 不得替换该路径
