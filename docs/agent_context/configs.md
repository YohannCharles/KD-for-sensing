# 配置任务上下文

tracked MMW recipe 是 `u0.yaml`、`amber_full.yaml`、`rmbp_mm.yaml` 和 `_base.yaml`。它们只有在 launcher 注入精确绑定的 clean protocol 后才可运行。DeepSense6G 保留 `configs/deepsense6g/t2.yaml`。

recipe 在没有数据、outputs、cache 和 checkpoint 的环境中必须能解析。不要新增旧 YAML alias、兼容字段或从本地产物读取配置。

先读 `openspec/specs/u0-mainline/spec.md` 与 `clean-data-integrity/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。
