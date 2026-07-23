# AI / Maintainer Navigation

当前维护面仅包括 Clean MMW U0、AMBER-Full、RMBP-MM、DeepSense6G Scene31--34 T2，以及 train、evaluate、preprocess 三个 package CLI。

非平凡改动先读 `AGENTS.md`、三个 current spec 和 `docs/maintainer_context_index.yaml`；随后只加载与任务匹配的一份 scoped context。MMW 数据或运行改动必须先确认 clean protocol，不能把 validation、confirmation 或 outer test 引入训练。

最小验证：

```bash
conda run -n kd_mm_beam pytest tests/test_clean_inner_protocol.py tests/test_u_mask_beam_jepa.py tests/test_deepsense6g_dataset.py -q
make verify-quick
```
