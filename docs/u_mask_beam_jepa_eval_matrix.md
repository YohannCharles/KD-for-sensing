# MMW Fixed-Mask Evaluation

MMW all-weather helper 对 T2、S1、AMBER-Full、RMBP-MM 使用共享 fixed masks 和样本身份。它是本地 evidence workflow，不是额外 public CLI；输出写入 `outputs/`。

mask 保证至少一个模态可用。比较必须固定 split、checkpoint epoch、mask identity 与指标定义；结果的 claim 状态由 `docs/result_claims_registry.md` 管理。
