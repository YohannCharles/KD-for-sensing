# 数据任务上下文

数据面服务 MMW prepared sequence 与 DeepSense6G Scene31–34 标准 CSV 的 image、radar、gps、lidar 输入及其共享时间缺失协议。DeepSense6G 标签只能来自 64 维 `future_beamN` 功率的 `argmax`；不恢复 CSI、毫米波原始输入、soft label、cache 或场景 alias。`dataset/` 是本地输入，不能纳入源码变更；训练输出、cache、日志和 checkpoint 也不是文档或配置输入。

先读 `openspec/specs/t2-baseline-surface/spec.md` 和 `openspec/specs/project-architecture/spec.md`。不要恢复任何已退役数据路线。

最小验证：`conda run -n kd_mm_beam pytest tests/test_deepsense6g_dataset.py tests/test_mmw_all_weather_runtime.py -q`。
