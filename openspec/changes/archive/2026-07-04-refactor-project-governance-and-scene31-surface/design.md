## Context

项目已经有 `project_surface_inventory`、OpenSpec lifecycle 和架构边界测试，但当前真实表面积再次增长：`configs/scene31/` 跟踪大量 local/manual YAML，Scene31 shell runner 复制了相似的 worker、skip、fresh eval 和 failed-list 逻辑，inventory 中的数量和“generated YAML 不长期入库”说明已经与实际状态不完全一致。

## Goals / Non-Goals

**Goals:**
- 将 Scene31 local/manual 实验面重新收敛为 manifest-backed surface。
- 更新 inventory、导航文档和架构边界测试，使真实 tracked 文件和 lifecycle 分类一致。
- 复用 Scene31 runner/generator/summary 的共同逻辑，减少重复 shell 或 Python glue。
- 保持已有本地运行路径、run name、输出 root 和 fresh eval 指标口径可追溯。

**Non-Goals:**
- 不启动或重跑 Scene31 训练。
- 不删除、移动、压缩或改写本地 `outputs/`、`logs/`、checkpoint。
- 不改变 U-MaskBeamJEPA、BTAPA、beamsoft、MP-DRO 或 adaptive sampler 的训练语义。

## Decisions

1. **先治理，再收敛实现。**
   先更新 inventory/OpenSpec/架构测试的事实基线，再改 generator/runner。这样可以把“哪些 YAML 必须保留，哪些应改为本地生成”变成可验证契约。

2. **Scene31 runner 共用逻辑优先抽到 Python helper 或窄 shell library。**
   现有 shell launcher 复制 GPU worker、状态检查和 failed list。共用逻辑必须只负责调度与 IO，不复制 `kd-sensing-train` 或 apples-to-apples eval 的业务实现。

3. **generated YAML 默认不作为长期源码资产。**
   对必须保留的 YAML，inventory 需要说明 owner、local/manual 理由和删除触发条件；否则使用 generator + manifest 在本地重建。

4. **completed change 状态先清晰化。**
   若相关 Scene31 change 已 complete，应归档或在新 change 中说明 deferral，避免后续维护者把它们当 active 需求。

## Risks / Trade-offs

- 误删仍被测试或用户使用的 Scene31 YAML -> 先用 manifest/generator sanity test 和 `tests/test_scene31_next_round.py` 固定 run name 与字段。
- 抽 runner 逻辑时改变 skip/overwrite 行为 -> 保留 shell smoke 或 Python helper unit test，覆盖 completed/skipped/failed/eval_failed。
- inventory 只更新文字但没有护栏 -> 同步更新 `tests/test_architecture_boundaries.py`，拒绝未登记脚本/YAML 回流。

## Migration Plan

1. 捕获当前 tracked Scene31 YAML、shell、generator 和 summary 清单。
2. 更新 inventory、OpenSpec delta 和架构测试期望。
3. 抽共享 runner/generator helper，保持现有 shell 命令入口可运行。
4. 将可再生成 YAML 从长期源码面移除或登记为 local/manual 保留。
5. 运行 `openspec validate refactor-project-governance-and-scene31-surface --strict`、`conda run -n kd_mm_beam pytest tests/test_scene31_next_round.py tests/test_architecture_boundaries.py -q`。

## Open Questions

- 哪些 Scene31 实体 YAML 需要作为人工样例长期保留，哪些只保留 manifest 行即可？
- 现有 shell launcher 是否全部保留为用户入口，还是改为一个统一 Python runner 加少量 shell alias？
