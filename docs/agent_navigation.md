# AI / Maintainer Navigation

当前支持面包含 MMW T2/baseline、DeepSense6G Scene31--34 T2、BCACL U2/CMSBL 训练 objective，以及训练、评估、预处理三个 package CLI。

非平凡改动先读 `AGENTS.md`、active OpenSpec 和 `docs/maintainer_context_index.yaml`。模型改动聚焦 T2、BCACL U2、CMSBL、BPA/CMA 与 MMW baseline；数据改动聚焦 MMW prepared sequence 或 DeepSense6G 标准 CSV 的四模态 sequence；脚本改动只聚焦 MMW all-weather、BPA/CMA、summary 和验证。

最小验证：

```bash
conda run -n kd_mm_beam pytest tests/test_deepsense6g_dataset.py tests/test_mmw_prepared_sequence.py tests/test_mmw_all_weather_runtime.py tests/test_cli_strict_overrides.py tests/test_s1_temporal_superset_training.py -q
make verify-quick
```

历史路线不再有兼容入口或 YAML migration。用途与追溯方式见 [retired_routes.md](retired_routes.md)。
