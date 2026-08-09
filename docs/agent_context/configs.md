# 配置任务上下文

canonical MMW recipe 是 `u0.yaml`、`amber_full.yaml`、`rmbp_mm.yaml` 和 `_base.yaml`；DeepSense6G 保留 `configs/deepsense6g/t2.yaml`。四模态 topology predictor 的唯一 local template 位于 `tools/configs/topology_predictor/`，运行入口只注入精确绑定的 MMW protocol、train seed 与 ULA-DFT topology audit。

tracked template 在没有数据、outputs、cache 和 checkpoint 的环境中必须能解析。不要新增历史 YAML alias、兼容字段或在 import/解析期读取本地产物。

先读 `openspec/specs/four-modal-topology-predictor/spec.md`、`u0-mainline/spec.md` 与 `clean-data-integrity/spec.md`。最小验证：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_four_modal_topology_predictor.py -q`。
