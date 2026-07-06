## Why

Scene31-34 主实验已经完成到论文可引用阶段，但现有归档 change 只覆盖训练、fresh eval、summary、missing-count 曲线、compute profile 和基础表格。当前还缺少论文最终分析证据包：显著性检验、pattern-level heatmap、误差 CDF、采样分布说明、最终表格整合和一键 final analysis runner。

本 change 不训练新方法，只读取已有 `outputs/scenes31_34_*` 结果，并把最终结论收敛到 `prototype + random non-empty subset exposure`。

## What Changes

- 新增显著性检验脚本，支持 seed-level paired test、sample/pattern bootstrap、per-scene delta 和 paper table 输出。
- 新增 pattern-level heatmap / delta / win-count 分析脚本。
- 完善 compute profile 输出字段和 paper table 写入。
- 新增 absolute beam error CDF 分析脚本。
- 新增 sampling distribution 统计脚本，解释 random subset exposure 与 Bernoulli randomdrop 的分布差异。
- 新增 final paper table updater 与一键 final analysis runner。
- 扩展最终结论脚本，读取 statistics、pattern analysis、profile、CDF 和 sampling artifacts，并保守写出论文结论。
- 最终 polish 阶段统一 significance delta / CI 单位口径，补轻量 inference latency benchmark，生成论文友好 degradation / heatmap / CDF 图，并新增 final polish runner。

## Impact

- 影响 `scripts/` 下 Scene31-34 final analysis、table update、profile 和 conclusion 脚本。
- 新增轻量测试 `tests/test_scene31_34_final_analysis.py`，使用 fixture，不读取真实 dataset、不训练模型。
- 生成 CSV/MD/PNG/PDF/TXT 仍写入 ignored `outputs/`，不纳入源码。
- 验证包括 `openspec validate promote-scenes31-34-main-missing-count --strict` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
