## ADDED Requirements

### Requirement: 旧实验和诊断表面必须可删除或降级
项目 MUST 对已审计为低价值的旧实验、旧诊断、孤岛 helper 和 local/manual runbook 建立删除或降级边界。候选项只有在不属于当前 package CLI、registry、canonical config、README/docs 当前入口、OpenSpec current requirement 或必要 focused test 输入时，才可删除；否则 MUST 降级为薄兼容 reader、local/manual surface 或记录删除触发条件。

#### Scenario: 删除旧 full sweep runner
- **WHEN** `cnn_hybrid_jepa_visual_prior_sweep` 的训练 runner、job graph、shell 生成和 cleanup 逻辑不再被 current docs、CLI、tests 或 OpenSpec 当前 requirement 需要
- **THEN** 本 change MUST 删除这些旧执行逻辑或将其降级为只读兼容 reader
- **AND** 当前推荐入口 MUST 指向 `jepa_visual_architecture_sweep` owner 和 manifest

#### Scenario: 删除孤岛 helper 前有证据
- **WHEN** CodeGraph 或结构检查显示某个 public helper 无当前内部调用
- **THEN** 删除任务 MUST 同时检查 pyproject、README/docs、OpenSpec current specs、tests、registry 和 package `__all__`
- **AND** 无公开契约消费时才可删除；否则 MUST 记录保留理由或降级计划

#### Scenario: 本地 runbook 不升级为当前入口
- **WHEN** 固定 GPU shell、一次性 runner 或本地 overlay YAML 只服务 Scene31/RBMA/M2Beam 本地实验
- **THEN** inventory MUST 将其标记为 local/manual、删除候选或已归档历史
- **AND** README quickstart 和 package CLI MUST 不把该脚本描述为长期推荐入口

### Requirement: Ignored bytecode 清理不得混同源码删减
项目 MAY 清理 ignored Python bytecode 和 cache 目录，但 MUST 将其作为工作区噪声清理，而不是源码行为变更。该清理 MUST 不删除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或历史权重。

#### Scenario: 清理 pycache 和 pyc
- **WHEN** 工作区存在 `__pycache__` 或 `.pyc` 文件
- **THEN** 本 change MAY 删除这些 ignored 本地文件
- **AND** 最终说明 MUST 明确它们不属于源码变更

#### Scenario: 实验产物仍受保护
- **WHEN** 候选路径位于 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/`
- **THEN** 本 change MUST 不自动删除该路径
- **AND** 若用户另行要求删除，流程 MUST 使用 runtime cleanup manifest 或单独显式确认

### Requirement: 清理结果必须同步 current surface 文档
项目 MUST 在清理旧实验表面时同步 `docs/project_surface_inventory.md`、相关 README/docs、OpenSpec lifecycle 和架构边界测试。删除、降级或保留的候选项 MUST 有 owner、替代路径、验证命令和回滚方式。

#### Scenario: 删除后引用一致
- **WHEN** 本 change 删除或降级源码、脚本或配置
- **THEN** README、docs、tests 和当前 OpenSpec specs MUST 不再把旧路径声明为 current 支持入口
- **AND** 若保留历史说明，文档 MUST 明确标记为历史、local/manual 或兼容 reader

#### Scenario: 保留项有删除触发条件
- **WHEN** 某个 local/manual 脚本或 overlay YAML 因本地实验仍可能运行而保留
- **THEN** inventory 或实现说明 MUST 记录保留理由、输出边界和未来删除触发条件
- **AND** 保留项 MUST 不新增兼容 wrapper 或通用抽象层
