## 1. 统计汇总

- [x] 1.1 定义 statistical summary、paired comparison、claim gate 和 warning schema。
- [x] 1.2 实现 CSV/JSON 输入读取，支持 Scene31 missing-pattern 和 fresh-eval summary。
- [x] 1.3 实现 mean/std/count/min/max、bootstrap CI、paired delta 和 win/loss/tie count。
- [x] 1.4 添加样本不足、seed 不配对和字段缺失 focused tests。

## 2. Stress Suite Manifest

- [x] 2.1 定义 missing-modality stress manifest schema，覆盖 model groups、conditions、severity、strict fields 和输出边界。
- [x] 2.2 实现 smoke/quick/formal manifest normalizer。
- [x] 2.3 定义 canonical condition taxonomy：full、single missing、multi missing、only modality、non-GPS-only、random missing、unavailable modality。
- [x] 2.4 添加 manifest validation tests，不读取真实 `dataset/`。

## 3. Difficulty 与 Eval Matrix 接入

- [x] 3.1 扩展 eval matrix 输出 comparability fields 和 pattern group metadata。
- [x] 3.2 在 difficulty pipeline 中标准化 missing-modality stress preset。
- [x] 3.3 增加 radar/LiDAR/mmWave unavailable 表达或 adapter。
- [x] 3.4 确认 stress transform 不改变 target、beam power、sample id 和 split metadata。

## 4. Baseline 与文档

- [x] 4.1 为 AMBER-lite/full、RMBP-MM、U-MaskBeamJEPA 等 baseline 定义 stress comparability metadata。
- [x] 4.2 更新主线文档和 claim registry 规则，说明统计/stress claim gate。
- [x] 4.3 运行 `openspec validate add-missing-modality-statistics-stress-suite --strict`。
- [x] 4.4 运行 `conda run -n kd_mm_beam pytest` 的 eval matrix、difficulty、statistics focused tests。
- [x] 4.5 检查 `git status --short --untracked-files=all`，确认未纳入真实 metrics、figures、checkpoint、cache、`dataset/` 或 `logs/`。
