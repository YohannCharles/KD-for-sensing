# 数据任务上下文

数据面只服务 MMW prepared sequence 的 image、radar、gps、lidar 输入及其时间缺失协议。`dataset/` 是本地输入，不能纳入源码变更；训练输出、cache、日志和 checkpoint 也不是文档或配置输入。

先读 `openspec/specs/t2-baseline-surface/spec.md` 和 `openspec/specs/project-architecture/spec.md`。不要恢复任何已退役数据路线。

最小验证：`conda run -n kd_mm_beam pytest tests/test_mmw_all_weather_runtime.py -q`。
