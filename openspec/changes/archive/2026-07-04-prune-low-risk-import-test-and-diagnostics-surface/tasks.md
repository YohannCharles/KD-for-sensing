## 1. Import/export 机械瘦身

- [x] 1.1 列出源码、脚本和测试中残留的 `from __future__ import annotations`、runtime star import 和内部 `__all__`。
- [x] 1.2 删除无语义价值的 future annotations import，并保持该修改与行为变更分离。
- [x] 1.3 将 runtime star import 改为显式 import 或删除无价值兼容模块。
- [x] 1.4 删除无 public 契约价值的内部 `__all__`，保留项记录理由。

## 2. 测试与第二梯队 diagnostics

- [x] 2.1 拆分 `tests/test_training_io_workflow.py` 中 dataset/cache/label/run metadata 相关测试。
- [x] 2.2 拆分 JEPA benchmark、GPS-conditioned JEPA 或 modality difficulty 的超大测试文件，使 focused tests 对应 owner。
- [x] 2.3 拆分 `diagnostics/run_index.py` 的 scanner、resource collector、artifact summary 和 writer。
- [x] 2.4 拆分 `diagnostics/runtime_artifact_cleanup.py` 的 scan rules、manifest render、delete/apply validation 和 organize planning。
- [x] 2.5 评估 `research_claim_harvester.py` 是否需要拆 writer/collector 或登记 accepted-size 理由。

## 3. 验证

- [x] 3.1 运行 `openspec validate prune-low-risk-import-test-and-diagnostics-surface --strict`。
- [x] 3.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_run_index.py tests/test_runtime_artifact_cleanup.py -q`。
- [x] 3.3 按实际拆分后的测试文件运行对应 focused tests，并在最终说明中列出替代命令。
