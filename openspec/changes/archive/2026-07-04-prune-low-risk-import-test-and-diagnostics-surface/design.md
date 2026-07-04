## Context

仓库已经完成多轮旧入口清理，但仍有低风险维护噪音：残留 future annotations import、运行时星号导入、较多内部 `__all__`、超大测试文件，以及 run index、runtime cleanup、research claim harvester 等第二梯队诊断热点。它们适合独立 change 处理，避免与训练/模型重构耦合。

## Goals / Non-Goals

**Goals:**
- 清理机械 import 噪音并保持轻量 import 边界。
- 拆分大测试文件，使测试与 owner 模块更对应。
- 拆分 run index、cleanup 和 research claim harvester 的 scanner/collector/writer 边界。
- 保持 architecture guard 对旧入口、tracked artifact 和 facade 回流的覆盖。

**Non-Goals:**
- 不改变 public CLI 或 package API。
- 不删除真实 runtime artifact。
- 不借清理机会移动 dataset/training/model 行为。

## Decisions

1. **机械清理单独提交/任务。**
   删除 future annotations、星号导入和无价值 `__all__` 应与行为重构分开验证，便于回滚。

2. **测试拆分不降覆盖。**
   拆大测试只改变文件组织，保留原断言和 fixture；必要时新增 focused import smoke。

3. **diagnostics 二级热点按 writer/collector 拆。**
   run index 和 cleanup 的输出 schema 是用户可见契约，优先拆 scanner、resource collector、manifest renderer/apply，不改字段。

## Risks / Trade-offs

- 删除 `__all__` 影响非推荐外部 import -> 只清理无 current public 契约的内部 owner，并在最终说明中标记 breaking 风险。
- 测试拆分遗漏 fixture -> 先跑拆分后的 focused tests，再跑架构边界。
- cleanup 行为风险高 -> 删除执行路径必须保留 `--delete --manifest --confirm-delete` 三重确认。

## Migration Plan

1. 列出现有 future import、star import、internal `__all__` 和大测试文件。
2. 机械清理 import/export 噪音。
3. 拆分大测试文件，保留原测试语义。
4. 拆 run index / cleanup / claim harvester 二级热点。
5. 运行 `openspec validate prune-low-risk-import-test-and-diagnostics-surface --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_run_index.py tests/test_runtime_artifact_cleanup.py -q`，再按拆分测试补充 focused tests。

## Open Questions

- 哪些 `__all__` 应保留为 public API，哪些只是历史镜像？
- 是否把 run index 和 cleanup 的 writer helper 共享，还是保持各自领域 owner？
