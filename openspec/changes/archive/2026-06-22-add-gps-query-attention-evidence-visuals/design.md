## Context

当前仓库已经具备三块基础：`GPSQueryPool` 可记录 `[B,T,K,N]` attention map，`kd-sensing-jepa-visual-analysis` 已有离线表格/图表/report 产物结构，`cnn_hybrid_jepa_visual_prior_sweep` 和 P0-P5 benchmark 已经产生 GPS-query、mean pooling、strong supervised baseline 的可比指标。缺口不是训练新模型，而是把这些信号整理成一套可复查的 GPS-query 有效性证据包。

注意力图只能作为解释性证据，不能单独证明因果有效性。因此方案必须同时绑定 paired ablation、扰动切片、失败案例和 claim gate，避免只展示成功样本的热点图。

## Goals / Non-Goals

**Goals:**

- 生成一个离线 evidence package，用于回答“GPS-query 是否有效、在哪些条件有效、注意力是否有合理视觉聚焦、哪些情况失败”。
- 输出论文/答辩可用的图表：paired metric bar/heatmap、Scene31 与 S32-S34 对照、query gain/regression case panel、GPS-query attention hotspot overlay、attention statistic distribution 和 claim gate summary。
- 保证产物可审计：所有图表能追溯到模型 config、checkpoint、split、condition、sample id、baseline 和 metric delta。
- 复用现有训练结果和 benchmark 输出，不启动训练，不修改 checkpoint。

**Non-Goals:**

- 不新增 GPS-query 训练结构、不改变 `GPSQueryPool` 主 forward 语义。
- 不把 attention/Grad-CAM 单独包装成因果证明。
- 不恢复旧 KD、Hist、Top8、GPS residual、camera residual、BGAM 或 viewer manifest 路线。
- 不要求提交真实 `outputs/`、cache、checkpoint、图表或本地数据。

## Decisions

### Decision 1: 证据包作为离线分析模式

实现优先放在 `src/kd_sensing/diagnostics/` 的窄 helper 或 `jepa_visual_analysis.py` 的 opt-in 分支中，输入为 analysis config、benchmark manifest 或现有 P0-P5 wide CSV。输出统一到 `outputs/analysis/gps_query_effectiveness_visualization/<run_id>/`。

理由：现有 visual analysis 已经有 manifest、tables、figures、report 和 attention fallback 结构，最少代码即可复用。备选方案是新增独立训练/评估 CLI，但会复制 dataset/model/metric 逻辑。

### Decision 2: 证明口径以 paired ablation 为主，attention 为辅

主证据按同 split、同 seed、同 metric、同 condition 聚合 paired delta：

- GPS-query vs mean pooling，例如 `pooler_gps_query_k2_tokens` 对 `pooler_mean`。
- 同视觉 encoder 的 query vs mean，例如 `patch16_gps_query_pool` 对 `patch16_mean_baseline`。
- GPS-query best vs strong non-JEPA/image+GPS baseline，只作为外部 anchor，不替代 paired ablation。

图表包括 `paired_delta_by_condition`、`p0_p5_delta_heatmap`、`scene_group_delta` 和 `claim_gate_summary`。只有 paired ablation 在 clean/P0 与 P0-P5 mean 上通过阈值、且没有严重 clean regression 时，报告才允许写“支持 GPS-query 有效”。

### Decision 3: 注意力热点图必须绑定样本和对照结果

attention 输出分三层：

- `patch_grid`: 现有 patch-grid 热图，便于检查 token reshape 是否正确。
- `image_overlay`: 将 grid resize 到输入图尺寸，叠加到反归一化图像，展示热点区域。
- `case_panel`: 同一 sample 展示原图、GPS-query overlay、mean baseline/GPS-query Top-k 曲线、target beam、DBA contribution、query gain/regression 标签。

每张图必须带 sample id、scene、condition、target beam、Top-k、metric delta、history frame 和 query 聚合方式。没有原始图像或反归一化信息时，降级为 patch-grid 和表格，不失败。

### Decision 4: case selection 必须 deterministic 且包含失败

case 选择从跨模型 comparison table 产生，固定 seed 和排序规则：

- `query_gain`: baseline 远错，GPS-query Top-3 命中或 DBA 明显提升。
- `query_regression`: GPS-query 相比 paired baseline 明显变差。
- `shared_near_miss`: 两者都接近 target，但 GPS-query 改善有限。
- `shared_failure`: 两者都远错或 P3/P4 严重下降。

报告默认每组最多输出少量代表样本，完整选择依据写入 `tables/case_selection.csv` 和 `cases/*.json`。

### Decision 5: 输出 claim gate，而不是自动写成功结论

`report.md` 单独给出三类结论：

- `reportable`: paired ablation 与扰动切片支持的结论。
- `interpretive`: attention hotspot、entropy、query diversity 等解释性发现。
- `caveat`: 单 seed、attention 非因果、Scene31/S32-S34 差异、强 baseline 仍可能接近或超过 GPS-query 的边界。

如果 paired delta 不稳定、attention 不可用、sample 数不足或 baseline 不可比，claim gate 标记为 `insufficient` 或 `exploratory`。

## Risks / Trade-offs

- [Risk] 注意力热点图被误解为因果证明。  
  -> Mitigation: report 和 manifest 将 attention 标记为解释性证据，claim gate 必须以 paired metric 为主。

- [Risk] 不同模型的 split、checkpoint selection 或 metric profile 不一致导致伪提升。  
  -> Mitigation: evidence manifest 记录 strict comparability 字段；不一致时拒绝进入 paired delta 表或标记不可比。

- [Risk] 原始图像路径、反归一化 profile 或 token grid metadata 缺失。  
  -> Mitigation: 降级到 patch-grid heatmap、attention summary 和 case 表，不中断其它证据。

- [Risk] 只选择 query gain 样本造成 cherry-pick。  
  -> Mitigation: deterministic case selection 必须同时输出 gain、regression 和 failure 组。

- [Risk] 现有 P0-P5 CSV 与 real-forward cache 字段不完全一致。  
  -> Mitigation: 先支持最小公共列；缺失逐样本 attention 时只输出 aggregate evidence，并在 manifest 记录缺失原因。

## Migration Plan

1. 新增 evidence package config 示例，指向当前 `cnn_hybrid_jepa_visual_prior_sweep` 的 P0-P5 CSV、benchmark manifest 和 GPS-query/mean baseline 模型。
2. 实现 paired delta、case selection、attention overlay helper 和 evidence manifest 写出。
3. 将报告接入现有 `kd-sensing-jepa-visual-analysis` 或新增薄 CLI 子入口；所有项目 Python 命令使用 `conda run -n kd_mm_beam ...`。
4. 增加 synthetic/mock tests，不读取真实 `dataset/`。
5. 回滚时删除新 opt-in config/helper；既有训练、评估和 benchmark 输出不受影响。

## Open Questions

- overlay 是否默认读取真实 image path，还是先只支持 forward batch 中已加载 image tensor的反归一化结果？
- claim gate 的最小阈值先固定为配置项，还是先只输出 delta 和人工判读？
- 是否需要在第一版支持多 seed 聚合，还是先把单 seed 明确标记为 exploratory？
