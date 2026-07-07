## Context

P2 候选有一个共同特点：静态看起来“可能过大或过期”，但 current specs/configs/tests 仍可能依赖。典型例子：

- `src/kd_sensing/data/loso.py` 静态 inbound 很少，但 `cross-scene-loso-workflow` 仍描述 LOSO planning helpers。
- `project_surface_doctor` 输出量大，尤其无问题时也能打印很长的 pass inventory。
- whole-model `token_transformer` 与 modular sequence token core 有重叠，但 configs/registry 可能仍引用旧实现。
- architecture summary/export 支持多格式输出，可能超过当前 guardrail 和 docs 的实际需要。

P2 因此采用“先证据、后删除”的退役审计策略。

约束：

- 所有 Python 验证命令 MUST 使用 `conda run -n kd_mm_beam <command>`。
- 不删除仍被 current config、spec、test 或 documented workflow 消费的实现。
- 不用 compatibility wrapper 掩盖删除；若不能删，记录 retained-with-reason。

## Goals / Non-Goals

**Goals:**

- 对规格依赖强的支持性表面建立清晰退役闸门。
- 让 surface doctor 默认输出更短、更像检查工具，而不是清单 dump。
- 在 configs/tests/specs 迁移完成后删除 whole-model token transformer 或明确保留理由。
- 将 LOSO helper 的当前价值讲清楚：当前使用则保留，无消费者则退役。

**Non-Goals:**

- 不重写跨场景 LOSO 实验设计。
- 不改变模型数学或默认 registry 选择。
- 不删除任何仍被 canonical config 使用的模型实现。
- 不把 surface doctor 拆成多个新工具。

## Decisions

### Decision 1: LOSO 先审计消费者

`kd_sensing.data.loso` 只能在无 current docs/spec/tests/config/script 消费者时删除。若 `cross-scene-loso-workflow` 仍是 current requirement，则 implementation MUST 要么更新该 spec 为 historical/future contract，要么保留 LOSO helper 并记录 retained-with-reason。

### Decision 2: surface doctor 默认 issue-only

project surface doctor SHOULD 在默认模式下只输出问题、摘要和必要 next action。完整 pass inventory、allowlist dump 或 machine-readable governance 表 MUST 通过显式 flag 请求。

理由：治理工具默认输出太长，会降低人实际运行它的意愿，也让“无问题”结果看起来像产物。

### Decision 3: token transformer 删除必须以 config 迁移为前置

whole-model token transformer 只有在 registry、configs、tests 和 docs 全部迁到 modular sequence/token core 后才可删除。若仍有 canonical config 或 paper-facing recipe 引用旧 key，P2 只能记录保留理由和迁移计划。

### Decision 4: architecture summary/export 只保留当前消费者需要的格式

模型架构 summary/export 如果只为历史审计保留多种格式，implementation SHOULD 收缩到启动摘要和当前 guardrail/docs 使用的最小格式。删除某个格式前 MUST 检查 docs/tests/CI 是否消费。

## Migration Plan

1. 记录 baseline：`git status --short`、候选文件入站引用、configs/registry/docs/tests 引用。
2. 对 LOSO helper 执行消费者审计，并更新 `cross-scene-loso-workflow` current spec 或保留理由。
3. 修改 surface doctor 默认输出为 issue-only，增加或确认 full inventory opt-in flag。
4. 审计 token transformer registry/config 使用；可迁移则迁到 modular owner，不可迁移则记录 retained-with-reason。
5. 审计 architecture summary/export 格式消费者，删除无当前消费者的格式。
6. 更新 inventory、guardrails 和 focused tests。

Rollback：若删除后发现 current consumer，恢复 owner 实现并记录 retained-with-reason；不得用新 alias 或 stub 暂时遮掩。

## Validation Plan

- `openspec validate p2-retire-supporting-governance-surfaces --strict`
- `openspec validate --all --strict`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- surface doctor focused tests。
- LOSO focused tests 或无引用检查。
- model registry/config smoke tests，例如 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`
- `conda run -n kd_mm_beam python scripts/verify_compile.py`

## Open Questions

无阻塞问题。实现阶段 P2 最可能出现 retained-with-reason：如果 LOSO 或 token transformer 仍有 current consumer，就先不删，避免用“瘦身”破坏可复现实验。
