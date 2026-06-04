## Context

论文 `https://arxiv.org/pdf/2603.15093v1` 使用 Multimodal-Wireless / MMW Town10 数据集验证 historical beam index + BGAM。用户希望先在 MMW 上实现，便于和论文的 TopK / normalized gain / LiDAR-guided beam prediction 结果对齐。

仓库当前已有 MMW Town GPS v2 adapter、prepared split、scene-specific label calibration、frame manifest、RSU/CAV LiDAR path 和 LiDAR BEV cache。DeepSense6G 版本可以保留为后续迁移或诊断，但本 change 的主实验和默认配置改为 MMW。

## Goals / Non-Goals

**Goals:**

- 在 MMW Town10 sunny scenes 上实现 GPS pseudo-history + LiDAR BGAM reranker。
- 使用 frozen MMW GPS v2 logits/probs 生成 Top8 candidates 和历史 pseudo beam label。
- 默认 `mapping_enabled`，并支持 MMW scene-specific mapping fingerprint 校验。
- 保证 target/query true beam label 只用于 final metrics、diagnostics 和 report。
- 输出可复现 artifacts：Top8 manifest、BGAM manifest、pseudo-history summary、predictions、summary、normalized gain diagnostics 和 run metadata。

**Non-Goals:**

- 不复刻论文中的 LLM/GPT2 主干。
- 不把历史真实 beam label 作为默认 BGAM 输入；oracle-history 仅作为显式 upper bound。
- 不把 DeepSense6G scenario31-34 作为本 change 的第一阶段主实验。
- 不提交本地 MMW 数据、运行输出、cache 或 checkpoint。

## Decisions

### 1. MMW GPS v2 logits 是 frozen prior

MMW Top8 manifest builder 读取 `mmw_town_gps_adapter_v2` 导出的 `gps_logits.npy`、`gps_prior_probs.npy` 和 `gps_logits_index.csv`。候选 beam 从 logits 重新计算，不从 `predictions.csv` 的 Top5 字段截断推导。

### 2. scene-specific mapping fingerprint 必须逐 scene 校验

MMW calibration 存在 scene overrides，因此四个 Town10 scene 可以有不同 fingerprint。Top8/BGAM builders 按 row 的 `scene` 解析 expected mapping metadata，逐 scene 验证 logits、predictions、support manifest 和 Top8 manifest；不能用单一全局 fingerprint 检查所有 MMW rows。

### 3. pseudo-history 按 trajectory-safe group 构造

MMW 默认 `history.group_keys=[scene, agent, split]`，并按 frame/timestamp/sample suffix 排序取 nearest-past window。这样不会把不同 vehicle agent、不同 scene 或 train/test split 的历史串在一起。每行写出 `history_pseudo_beams`、prob、entropy、valid mask、timestamps、source row indices 和 missing count。

### 4. LiDAR 默认取 RSU/BS-side path

为贴近论文，MMW manifest 优先从 frame manifest 的 `rsu_json.agents.*.lidar` 取 RSU LiDAR；缺失时可回退到 prepared split 中的 CAV window LiDAR。BEV cache path 由当前 LiDAR preprocessing hash 生成；cache 缺失时可按配置重建。

### 5. normalized gain 是 MMW 报告字段

Top8 manifest 读取 future beam power path，并用 scene-specific inverse mapping 将 mapped beam 转回 raw beam index，计算 `gps_normalized_gain`、candidate normalized gain 和 Top8 oracle normalized gain。BGAM predictions/summary 透传 `final_normalized_gain` 与 delta vs GPS。

## Implementation Sketch

1. 新增 `configs/mmw_town_top8_selector.yaml` 和 `configs/mmw_town_gps_lidar_bgam.yaml`。
2. 扩展 MMW GPS v2 output metadata/support manifest，使 logits index、predictions、support manifest 记录 label-space 和 fingerprint。
3. 新增 `mmw_town_topk_candidate_manifest`：读取 frozen GPS logits，逐 scene 校验 fingerprint，合并 prepared CSV/frame manifest/LiDAR/beam power 信息，写 Top8 manifest。
4. 新增 `mmw_town_gps_lidar_bgam_manifest`：合并 MMW Top8、RSU LiDAR、pseudo-history 和 BEV metadata，写 `pseudo_history_summary.csv`。
5. 复用通用 `GPSLidarBGAMDataset`、`GPSGuidedBGAM` 和 `GPSLidarBGAMBeamPredictor`，保持 BGAM mask 不接触 target/query label。
6. 新增 MMW BGAM CLI/runner wrapper，并让通用 BGAM runner 支持无 support-ratio 子目录的 MMW 输出结构。
7. 更新 evaluator summary/predictions，加入 normalized gain 字段。
8. 更新 README、tests 和 OpenSpec tasks。

## Risks / Trade-offs

- [Risk] MMW GPS pseudo label 本身误差大，history mask 会放大错误区域 → Mitigation: entropy/adaptive sigma、TopK union、GPS-only baseline 和 oracle-history upper bound。
- [Risk] scene-specific fingerprint 混用导致指标不可解释 → Mitigation: Top8/BGAM manifest builder 按 scene 早失败。
- [Risk] RSU LiDAR BEV cache 不完整 → Mitigation: manifest 标记 `lidar_source` 与 missing reason；配置允许 cache 缺失时从 raw `.pcd` 重建。
- [Risk] 本地 l5p3 split 与论文完整预测设置不完全一致 → Mitigation: config 显式记录 split tag、history_len 和 prediction_horizon，comparison report 必须说明对齐差异。

## Migration Plan

1. 先重跑 MMW GPS v2：`--save-logits --save-prior-probs`，得到 `outputs/analysis/mmw_town_gps_adapter_v2/mapping_enabled/gps_logits.npy` 和 index。
2. 运行 MMW Top8 manifest builder 和 BGAM manifest enrich，检查 pseudo-history coverage、fingerprint 和 LiDAR availability。
3. 跑 MMW small smoke，再跑完整 Town10 mapped 主实验。
4. 单独跑 `mapping_disabled` raw-label 对照，比较 mapped vs raw 的 history continuity、BGAM DBA、circular error 和 normalized gain。
5. 更新 comparison report，说明与 arXiv:2603.15093v1 的同点、差异和后续调参建议。
