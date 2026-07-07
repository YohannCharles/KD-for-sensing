## 1. Baseline 与消费者审计

- [x] 1.1 运行 `git status --short`，确认本 change 不包含本地数据、outputs、logs、cache、checkpoint 或历史权重。
- [x] 1.2 枚举 LOSO、surface doctor、token transformer、architecture summary/export 的 docs/spec/tests/config/registry/script 引用。
- [x] 1.3 为每个候选标注 `delete`、`shrink-default`、`migrate-then-delete` 或 `retained-with-reason`。

## 2. LOSO helper 退役或保留说明

- [x] 2.1 审计 `kd_sensing.data.loso` 的 current consumers，包括 OpenSpec、docs、tests、scripts 和 configs。
- [x] 2.2 不适用：current `cross-scene-loso-workflow` 仍消费 fold/few-shot supporting 语义，因此不删除 LOSO helper。
- [x] 2.3 若仍有 current consumer，保留 helper，记录 retained-with-reason、owner 和未来删除触发条件。
- [x] 2.4 运行 LOSO focused tests 或无引用检查。

## 3. Surface doctor 默认输出瘦身

- [x] 3.1 将 project surface doctor 默认输出改为 issue-only 摘要和必要 next action。
- [x] 3.2 增加或确认显式 full inventory flag，例如 `--dump-inventory`。
- [x] 3.3 更新 docs/tests，使无问题时默认输出不会 dump 大段 pass 清单。

## 4. Token transformer 迁移审计

- [x] 4.1 审计 whole-model token transformer 的 registry、config、docs 和 tests 引用。
- [x] 4.2 不适用：仍有 current entity configs、registry/tests 和 canonical CLS-token lightweight consumer，不能迁移后删除。
- [x] 4.3 若仍有 current consumer，保留实现并记录 retained-with-reason 与迁移触发条件。
- [x] 4.4 运行 model registry/config smoke tests。

## 5. Architecture summary/export 瘦身

- [x] 5.1 审计 architecture summary/export 的格式消费者。
- [x] 5.2 不适用：JSON、Markdown 和 CSV 均有 current docs/tests/CLI 或 paper-facing consumer。
- [x] 5.3 保留启动摘要、必要 guardrail 和 docs 当前需要的最小输出。

## 6. 验证与收尾

- [x] 6.1 更新 `docs/project_surface_inventory.md`、project hotspot/governance docs 和 architecture guardrails。
- [x] 6.2 运行 `openspec validate p2-retire-supporting-governance-surfaces --strict`。
- [x] 6.3 运行 `openspec validate --all --strict`。
- [x] 6.4 运行 architecture、surface doctor、LOSO 或无引用、model config/registry focused validation（architecture/CLI help 存在既有失败，见最终说明）。
- [x] 6.5 最终说明列出删除项、缩小默认输出项、retained-with-reason 和未运行验证。
