# AI / Maintainer Navigation

当前支持面仅为 MMW T2/baseline：`configs/mmw/t2.yaml`、`s1.yaml`、`amber_full.yaml`、`rmbp_mm.yaml`，以及训练、评估、预处理三个 package CLI。

非平凡改动先读 `AGENTS.md`、active OpenSpec 和 `docs/maintainer_context_index.yaml`。模型改动聚焦 T2、BPA/CMA 与 baseline；数据改动聚焦 MMW image/radar/gps/lidar sequence；脚本改动聚焦 all-weather matrix、T2 screening 和 BPA/CMA evidence。

最小验证：

```bash
conda run -n kd_mm_beam pytest tests/test_mmw_all_weather_runtime.py tests/test_s1_temporal_superset_training.py -q
make verify-quick
```

历史路线不再有兼容入口或 YAML migration。用途与追溯方式见 [retired_routes.md](retired_routes.md)。
