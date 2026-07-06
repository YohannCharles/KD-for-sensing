## Context

缺失模态主线已经完成脚本层面的第一轮收口。配置表面仍然很大：`configs/scene31/`、RBMA missing workflow、strong encoder overlay、JEPA image+GPS 派生实验和若干 local/manual seed sweep 都保留了大量实体 YAML。实体 YAML 的价值不一样：有些是 current claim/evidence 输入，有些是 paper reproduction，有些只是 generator 可重建的历史队列配置。

本 change 的关键不是“删最多”，而是让每个保留 YAML 都有理由，让可生成配置回到 generator/manifest/base config，让 docs/claims/tests 不再引用已经删除或历史化的路径。

约束：

- 所有项目 Python 验证命令 MUST 使用 `conda run -n kd_mm_beam <command>`。
- 不删除、不移动、不重写 `outputs/`、`logs/`、checkpoint、cache 或真实数据。
- 不让 virtual config 或 generator 接管 retired route。
- 不改变训练数学语义、metric 口径、split、label-space 或 checkpoint schema。

## Goals / Non-Goals

**Goals:**

- 对 Scene31、RBMA/KD/BTAPA、strong encoder 和 JEPA image+GPS config families 建立生命周期分类。
- 删除可由 generator/manifest/base config 无损重建且无 current evidence 依赖的实体 YAML。
- 保留 current claim、paper/workflow reproduction、diagnostic manifest 和必要 local/manual YAML，并记录理由。
- 补齐 generator focused tests 和 config doctor 分类。
- 更新 experiment matrix、mainline catalog、claim registry 和 inventory 引用。

**Non-Goals:**

- 不实现新的模型、训练策略或缺失模态方法。
- 不把所有实验 YAML 都迁成 virtual config。
- 不 archive 或搬运本地 outputs。
- 不删除仍被 claim/evidence/current docs/spec/tests 消费的配置。
- 不恢复旧 KD、BGAM、Hist、viewer、AMR mock 或 JEPA-MSAC 路线。

## Decisions

### Decision 1: 配置删除必须有等价或无依赖证据

实体 YAML 只有满足以下至少一项才可删除：

- generator/manifest/base config 能无损重建关键 resolved semantics。
- 无 current docs/spec/tests/claim provenance 引用，且只服务已沉淀历史结论。
- 已被明确更 canonical 的 config 或 recipe 取代。

理由：配置是复现实验的入口，不能只按重复文件名或目录规模删除。

替代方案：按目录批量删除旧 YAML。这样最快，但最容易破坏 claim provenance。

### Decision 2: 先处理 Scene31 与 RBMA，再处理 JEPA image+GPS

优先级：

1. `configs/scene31/`
2. `configs/fusion/experiments/rbma_missing_workflow*`
3. `configs/fusion/experiments/jepa_image_gps/`

理由：Scene31/RBMA 是当前缺失模态主线旁边的最大本地配置表面；JEPA image+GPS 仍有 secondary/supporting 价值，应该在第一批删除稳定后再缩。

替代方案：三类一起删。范围更大，但冲突和误删风险也更大。

### Decision 3: 保留 generator，不保留可再生成实体 YAML

对于规则化 seed sweep、missing pattern matrix、encoder ablation 或 local queue 配置，长期 source surface 优先保留 generator、manifest、base config 和 focused sanity test。实体 YAML 只有在作为 claim/evidence input、paper reproduction 或人工样例时保留。

理由：生成器是小而可审查的源，成批 YAML 是噪声。

### Decision 4: docs 和 claims 是删除门

删除或生成化配置前，必须检查并更新：

- `docs/project_surface_inventory.md`
- `docs/mainline_model_catalog.md`
- `docs/experiment_matrix.md`
- `docs/result_claims_registry.md`
- 当前 OpenSpec specs
- focused tests 和脚本默认路径

理由：配置删除最常见的回归不是代码失败，而是文档继续指向不存在路径。

## Risks / Trade-offs

- 删除实体 YAML 可能破坏私人复跑路径 → 保留 generator/manifest/base config，并在 inventory 给出复跑命令。
- 保留过多 evidence YAML 会让行数下降不明显 → 本 change 接受这一点；目标是可审计，不是追求最大删除数。
- generator 等价测试可能只覆盖关键字段而非字节级一致 → 明确允许 run identity 字段差异，但必须覆盖行为字段。
- 与 public entrypoint change 并行时 docs 冲突 → 两个 change 都可能改 inventory；合并时以 lifecycle 分类为准，不互相覆盖。

## Migration Plan

1. 生成 tracked YAML family inventory，标记 current/evidence/generated/local/manual/historical/delete。
2. 对每个 delete-candidate 检查 docs/spec/tests/scripts/claim 引用。
3. 为可再生成配置补 generator focused tests。
4. 删除或降级实体 YAML，保留 generator/manifest/base config。
5. 更新 docs、claim registry、experiment matrix 和 project surface inventory。
6. 运行 OpenSpec、config load characterization、generator tests、architecture boundary 和 config/surface doctor。

Rollback：若删除误伤复跑路径，恢复实体 YAML 并登记为 evidence/local-manual；不得通过 virtual alias 接管 retired 或 historical 路径。

## Open Questions

无阻塞问题。实现阶段若某个 YAML 是否仍支撑 claim 不确定，默认保留并登记删除触发条件。
