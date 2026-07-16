# 配置任务上下文

唯一 canonical inputs 是 `configs/mmw/t2.yaml`、`s1.yaml`、`amber_full.yaml`、`rmbp_mm.yaml` 和共享 `_base.yaml`。它们必须在没有 `outputs/`、数据和 checkpoint 的环境中解析。

先读 `openspec/specs/canonical-config-resolution/spec.md` 与 `openspec/specs/t2-baseline-surface/spec.md`。退役 YAML 直接不存在，不添加替代 recipe、fallback 或迁移层。

最小验证：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。
