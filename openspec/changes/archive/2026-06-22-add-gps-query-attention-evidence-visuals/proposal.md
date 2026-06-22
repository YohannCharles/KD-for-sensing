## Why

现有 P0-P5 表能说明 GPS-query 候选在若干消融上优于 mean pooling，但还缺少一套可审计的可视化证据，把“指标提升”“注意力是否随 GPS 条件聚焦到有效视觉区域”“失败案例边界”放在同一报告中。现在需要为论文和答辩准备 claim-ready 的图表，而不是只展示单张好看的注意力图。

## What Changes

- 新增一个离线 GPS-query 有效性证据包方案，读取已训练模型、已有 benchmark/forward cache 或现有评估配置，输出 paired ablation 表、query gain/regression case、注意力热点图和 claim gate 报告。
- 为 GPS-query 模型导出 attention heatmap，包括 patch-grid 热图、叠加到原始/反归一化图像上的 overlay、按 history frame/query 聚合的面板，以及 attention entropy、effective patch count、query diversity、center-of-mass 等统计。
- 将注意力图与 paired baseline 对齐：至少比较 GPS-query 与 mean pooling、同视觉 encoder 的 patch16 对照、非 JEPA strong baseline，区分 clean/P0、P1-P5、Scene31 泛化和 S32-S34 seen-scene 总体。
- 增加 deterministic case selection：`query_gain`、`query_regression`、`shared_near_miss`、`shared_failure`，禁止只挑成功案例。
- 输出机器可读 `evidence_manifest.json`、`tables/`、`figures/`、`cases/` 和 `report.md`；所有真实图表、cache 和 case payload 仍写入 ignored 的 `outputs/`。
- 不新增训练路线、不修改 checkpoint、不恢复旧 KD/Hist/residual/BGAM 等退役入口。

## Capabilities

### New Capabilities

- `gps-query-effectiveness-visualization`: 定义 GPS-query 有效性可视化证据包的输入、paired comparison、attention hotspot、case study、claim gate 和产物边界。

### Modified Capabilities

- 无。

## Impact

- 主要影响 `src/kd_sensing/diagnostics/jepa_visual_analysis.py` 或一个窄的同级 helper 模块，以及对应 CLI/config 中的 opt-in 分析配置。
- 复用现有 `GPSQueryPool` attention diagnostics、JEPA visual analysis、GPS shortcut benchmark、CNN/hybrid sweep summary 和 P0-P5 benchmark 输出。
- 新增或更新 focused tests，覆盖 attention reshape/overlay schema、paired gain 计算、case selection、attention unavailable 降级和 manifest 输出。
- 不新增强制依赖；图像 overlay 优先使用现有 matplotlib/Pillow/numpy 能力，缺失可选能力时记录 warning 并保留表格证据。
