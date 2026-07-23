# 模型任务上下文

`u_mask_beam_jepa` 是 Clean MMW U0 的四模态 masked-mean supervised router；它保留 prototype/BPA 和同模型 superset consistency。`modular_sequence` 保留 AMBER-Full 与 RMBP-MM。DeepSense6G 使用独立 T2 recipe，但共享四模态 batch contract。

不要恢复 BCACL、CMSBL、capacity、recovery、旧消融或 compatibility route。先读 `openspec/specs/u0-mainline/spec.md`。

最小验证：`conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py tests/test_amber_full_architecture.py tests/test_component_registry.py -q`。
