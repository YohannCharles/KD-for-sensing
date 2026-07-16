# Runtime 与证据任务上下文

训练与评估只服务固定 40 epoch、`last.pth`、MMW 15-domain 的 T2/baseline protocol。fixed-mask、多 seed、BPA/CMA 与 hyperparameter screening 的脚本只产生本地 evidence，不自动升级 claim。

先读 `openspec/specs/training-evaluation-runtime/spec.md`、`openspec/specs/mmw-baseline-multiseed-robustness-evidence/spec.md` 和相应 active change。不要恢复任何额外 diagnostic surface。

最小验证：`conda run -n kd_mm_beam pytest tests/test_mmw_all_weather_runtime.py tests/test_mmw_t2_hyperparameter_screening.py -q`。
