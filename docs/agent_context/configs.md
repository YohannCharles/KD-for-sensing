# 配置任务上下文

canonical MMW recipe 是 `u0.yaml`、`amber_full.yaml`、`rmbp_mm.yaml` 和 `_base.yaml`；DeepSense6G 保留 `configs/deepsense6g/t2.yaml`。PCPF-T local templates 位于 `tools/configs/pcpf/`，必须复用 config loader，并由运行入口注入精确绑定的 MMW protocol、stage checkpoint 和 gate provenance。

tracked template 在没有数据、outputs、cache 和 checkpoint 的环境中必须能解析。不要新增旧 YAML alias、兼容字段或在 import/解析期读取本地产物；PCPF sparse-CSI 配置只在显式本地运行时绑定 ignored cache。

先读 `openspec/specs/u0-mainline/spec.md` 与 `clean-data-integrity/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。
