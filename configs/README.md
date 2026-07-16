# Config Surface

当前源码配置只保留 MMW T2 主线及其可比较 baseline。

| Family | Paths | Purpose |
| --- | --- | --- |
| T2 / S1 / AMBER-Full / RMBP-MM | `configs/mmw/{_base,t2,s1,amber_full,rmbp_mm}.yaml` | 训练、评估和 all-weather matrix 的 tracked 输入 |
| MMW preparation | `configs/preprocess/mmw_radar_maps_all_weather.yaml` | Town03 all-weather 雷达图准备 |

历史 C2、DeepSense6G、CSI、physics、GPS-only、pretraining 和 H5/P1 配置已退役。历史用途见 `docs/retired_routes.md` 与 OpenSpec archive；不要恢复兼容 YAML、alias 或 virtual config。
