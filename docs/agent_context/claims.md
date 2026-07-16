# Claim 任务上下文

MMW 比较 claim 只能来自 tracked MMW recipe、固定 40 epoch `last.pth`、一致 split/mask identity 和对应 summary。DeepSense6G 尚无 tracked comparative claim；未来 claim 必须使用其独立 recipe、scene、split 与 mask provenance，不能与 MMW matrix 混合。历史结果不参与比较或升级。

先读 `docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和 `openspec/specs/mmw-baseline-multiseed-robustness-evidence/spec.md`。没有完整多 seed 结果时保持 `pending`，不得把 development screening 当成正式结论。
