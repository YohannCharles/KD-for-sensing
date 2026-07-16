# Claim 任务上下文

只有来自 tracked MMW recipe、固定 40 epoch `last.pth`、一致 split/mask identity 和对应 summary 的证据才可登记为当前 claim。T2、S1、AMBER-Full、RMBP-MM 以外的历史结果不参与比较或升级。

先读 `docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和 `openspec/specs/mmw-baseline-multiseed-robustness-evidence/spec.md`。没有完整多 seed 结果时保持 `pending`，不得把 development screening 当成正式结论。
