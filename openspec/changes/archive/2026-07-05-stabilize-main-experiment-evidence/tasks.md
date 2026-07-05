## 1. Scene31-34 evidence checklist

- [x] 1.1 梳理 Scene31-34 final evidence checklist：core proto n=5、classifier、external-lite、fresh eval、missing-count、per-scene、compute、paper tables、conclusion。
- [x] 1.2 更新 summary/export/conclusion scripts，使缺 baseline 或 mask_suspect 时输出 pending/incomplete caveat。
- [x] 1.3 增加或更新 smoke/py_compile 验证，覆盖 final summary 和 paper table scripts。

## 2. JEPA real benchmark gate

- [x] 2.1 定义 real benchmark manifest 的 required fields。
- [x] 2.2 更新 shortcut/predictive benchmark 读取逻辑，缺 checkpoint 或 comparability 字段时标记 unavailable/not_comparable。
- [x] 2.3 增加 focused tests 覆盖 smoke 不可升级、缺 checkpoint、split/metric mismatch 和 real candidate gate。

## 3. Claim / 文档同步

- [x] 3.1 更新 claim registry 中 Scene31-34 和 JEPA predictive/shortcut 的缺失 evidence 与升级条件。
- [x] 3.2 更新 mainline model catalog、experiment protocols、experiment history 和 experiment matrix 的必要行。
- [x] 3.3 确认真实 outputs、figures、metrics、checkpoint 和 cache 未进入源码变更。

## 4. 验证

- [x] 4.1 运行 `openspec validate stabilize-main-experiment-evidence --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam python -m py_compile scripts/summarize_scenes31_34_main.py scripts/plot_missing_count_degradation.py scripts/profile_scenes31_34_methods.py scripts/export_scenes31_34_main_paper_tables.py scripts/write_scenes31_34_main_conclusion.py`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py tests/test_missing_modality_stress.py -q`。
