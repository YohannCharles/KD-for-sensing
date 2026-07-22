# 配置任务上下文

canonical inputs 是 `configs/mmw/` 下的 T2、S1、AMBER-Full、RMBP-MM recipe，以及 `configs/deepsense6g/t2.yaml` 和各自 shared base。它们必须在没有 `outputs/`、数据和 checkpoint 的环境中解析。

先读 `openspec/specs/canonical-config-resolution/spec.md` 与 `openspec/specs/t2-baseline-surface/spec.md`。DeepSense6G canonical recipe 仅允许整数 Scene31--34 和四模态 64 类配置。BCACL/CMSBL 默认关闭，clean clone parse 不得读取 capacity stats、outputs 或 checkpoint；退役 YAML 直接不存在。

最小验证：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。
