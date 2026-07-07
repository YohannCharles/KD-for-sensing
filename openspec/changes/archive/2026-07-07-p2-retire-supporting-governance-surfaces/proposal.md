## Why

P2 对应“可能能删，但规格依赖更强”的支持性和治理性表面。审计候选包括 `kd_sensing.data.loso`、project surface doctor 的冗长默认输出、whole-model token transformer、模型架构 summary/export 支持面等。它们的风险不是实现难，而是 current specs、configs、tests 或协作者工作流可能仍把它们当契约。

因此 P2 不做鲁莽删除，而是建立退役审计 change：先证明无当前消费者或完成迁移，再删除；若仍有消费者，就缩小默认行为、补 retained-with-reason，并设置下一次删除触发条件。

## What Changes

- 对 `kd_sensing.data.loso` 做消费者审计；若 current workflow 不再消费，退役 LOSO helper 并将 `cross-scene-loso-workflow` 降为 historical/future contract；若仍消费，记录保留理由和 owner。
- 将 project surface doctor 默认输出改为 issue-only；完整 inventory dump 必须显式 `--dump-inventory` 或等价 flag，避免无问题时输出大段 pass JSON。
- 对 whole-model token transformer fusion 做配置/注册迁移审计；只有在 configs、tests 和 specs 已迁到 modular sequence/token core 后，才删除旧 whole-model 文件。
- 对模型架构 summary/export 支持面做瘦身审计；只保留启动摘要、必要 guardrail 和当前 docs 需要的格式。
- 更新 project hotspot/governance specs、inventory、architecture tests 和 focused validation。

## Capabilities

### New Capabilities

- 无。本 change 只退役或收缩支持性表面。

### Modified Capabilities

- `cross-scene-loso-workflow`：增加 LOSO helper 退役审计规则，允许无 current consumer 时降级为 historical/future contract。
- `project-health-guardrails`：调整 surface doctor 默认输出为 issue-only，完整清单显式 opt-in。
- `project-hotspot-governance`：增加 P2 支持面退役闸门和 retained-with-reason 要求。
- `model-architecture-extension-contract`：增加 whole-model token transformer 迁移到 modular owner 后才能删除的规则。

## Impact

- 影响范围：`src/kd_sensing/data/loso.py`、`src/kd_sensing/diagnostics/project_surface_doctor.py`、token transformer whole-model 实现与注册/config 引用、模型架构 summary/export、project inventory、architecture/hotspot guardrails 和相关 specs。
- 不影响范围：当前训练结果、数据产物、runtime artifacts、正式 claims、未迁移 config 的可运行性。
- 兼容性：P2 删除项必须先通过消费者审计；若有 current consumer，不删除，只缩小默认行为或记录保留理由。
- 验证：OpenSpec strict validate、architecture boundary、surface doctor focused tests、LOSO focused tests 或退役引用检查、model registry/config smoke tests。
