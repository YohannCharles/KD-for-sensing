## Why

仓库还有一批低风险但持续制造噪音的表面：源码中残留少量 `from __future__ import annotations`、1 处运行时星号导入、大量内部 `__all__`、超大测试文件，以及 run index / cleanup 等次级诊断热点。它们适合单独做机械瘦身和测试拆分，避免掺入高风险训练语义变更。

## What Changes

- 删除或保留有理由的 `from __future__ import annotations`，并将该机械修改与行为修改隔离。
- 将运行时星号导入改为显式 import，收缩无 public 契约价值的内部 `__all__`。
- 拆分过大的测试文件，使 focused tests 与 owner 模块对应，减少单文件维护成本。
- 拆分 run index、runtime artifact cleanup、research claim harvester 等次级诊断热点的 scanner、collector、writer、manifest apply/render 边界。
- 保持 package import、CLI help、architecture boundary、cleanup dry-run/delete confirm 和 run index output schema 兼容。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `project-import-surface-consolidation`: 增加星号导入、内部 `__all__` 和 package marker 的收缩要求。
- `project-hotspot-governance`: 更新低风险 mechanical cleanup、测试拆分和第二梯队 diagnostics 热点治理。
- `experiment-run-index`: 明确 scanner、resource collection、artifact summary 和 writer 的拆分边界。
- `runtime-artifact-cleanup`: 明确 manifest/apply/render/organize 拆分边界和删除确认兼容要求。
- `project-health-guardrails`: 保持架构边界测试覆盖旧入口回流、tracked runtime artifact 和 current path/config 引用。

## Impact

- 影响源码：`src/kd_sensing/eval/missing_patterns.py`、包内 owner 模块的 `__all__`、`diagnostics/run_index.py`、`diagnostics/runtime_artifact_cleanup.py`、`diagnostics/research_claim_harvester.py`。
- 影响测试：`tests/test_training_io_workflow.py`、`tests/test_jepa_gps_shortcut_benchmark.py`、`tests/test_gps_conditioned_jepa.py`、`tests/test_architecture_boundaries.py`、`tests/test_run_index.py`、`tests/test_runtime_artifact_cleanup.py`。
- 不改变 public CLI、运行产物目录、本地清理确认流程或测试 fixture 语义。
