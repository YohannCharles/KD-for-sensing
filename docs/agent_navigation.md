# AI / Maintainer Navigation

当前唯一 active research mainline 是 PCPF-T。稳定维护面还包括 U0、AMBER-Full、RMBP-MM、DeepSense6G Scene31--34 T2 与正式 MMW protocol；public CLI 仍只有 train、evaluate、preprocess 三个。

非平凡改动先读 `AGENTS.md`、四个 current spec、PCPF-T active change 和 `docs/maintainer_context_index.yaml`；随后只加载与任务匹配的一份 scoped context。MMW 数据或运行改动必须绑定 clean-inner 或 trajectory-disjoint protocol，不能把 validation、confirmation 或 outer test 引入训练状态。

最小验证：

```bash
conda run -n kd_mm_beam pytest tests/test_pcpf_temporal_transformer.py tests/test_pcpf_risk.py tests/test_u_mask_beam_jepa.py tests/test_deepsense6g_dataset.py -q
make verify-quick
```
