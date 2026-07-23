# 数据任务上下文

MMW 只能通过 `clean_protocol.py` 的 `inner_train` / `inner_validation` 数据域运行；所有 train-domain 与 validation-domain 配对都必须完成零重叠审计，outer test 不得访问。归一化只从 train 拟合。

DeepSense6G 保持 Scene31--34、四模态和 64 类 future-beam 契约，不使用 MMW protocol。`dataset/`、`outputs/`、cache、日志和 checkpoint 均为本地边界。

先读 `openspec/specs/clean-data-integrity/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_clean_inner_protocol.py tests/test_train_only_normalization.py tests/test_deepsense6g_dataset.py -q`。
