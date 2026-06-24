## 1. 现有路径复核

- [x] 1.1 复核 `kd-sensing-jepa-gps-shortcut-benchmark` 的 manifest normalization、real-forward cache、CxD suite 和 GPS-query advantage slice 调用路径
- [x] 1.2 复核当前 `cnn_hybrid_jepa_visual_prior_sweep` 输出中可复用的 config、weights、model group、split、seed 和 metric profile 字段
- [x] 1.3 确认实现不新增训练入口、不改写 checkpoint、不提交 `outputs/`、cache、日志或权重产物

## 2. Reused-weight 诊断 profile

- [x] 2.1 增加 reused-weight fusion diagnostic profile/preset，默认引用小型 CxD 条件集和 hard-negative A-slice
- [x] 2.2 复用现有 Scenario C、Scenario D、CxD 和 predictive advantage condition normalization，避免新增重复 condition parser
- [x] 2.3 在 manifest/provenance 中记录 config、weights、checkpoint provenance、split、seed、label space、metric profile、difficulty digest 和 comparability keys

## 3. 指标聚合

- [x] 3.1 输出 condition-level DBA、Top-1、Top-3、Top-5、clean delta、relative drop、sample_count 和 comparability status
- [x] 3.2 输出 paired baseline margin 表，覆盖 GPS-only、image-only、mean-pooling、GPS-query 和 supervised fusion 等可比模型组
- [x] 3.3 实现 `image_rescue`、`gps_rescue`、`fusion_interaction` 派生指标，并在缺少必要条件时标记 unavailable/not-comparable
- [x] 3.4 记录 hard-negative peer pool、beam offset constraint、fallback count，并让 fallback 过高的行不得升级 claim

## 4. Report 和 claim gate

- [x] 4.1 更新 report/summary，明确 P0-P5 是兼容鲁棒性表，CxD/A-slice 是融合机制诊断表
- [x] 4.2 调整 Predictive Robustness claim gate：P0-P5 margin 只能升级 P-suite claim，融合机制 claim 需要 reused-weight diagnostic metrics
- [x] 4.3 将新增 CSV/JSON/可选图表路径写入 machine-readable manifest

## 5. 测试与验证

- [x] 5.1 新增 focused tests 覆盖 reused-weight diagnostic manifest 解析、默认 condition set、派生指标公式和 not-comparable/fallback 处理
- [x] 5.2 使用 `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py tests/test_jepa_visual_analysis.py -q` 或更窄相关测试验证诊断路径
- [x] 5.3 使用 `conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark ...` 对 2-3 个现有 checkpoint 跑小型 reused-weight diagnostic smoke，并确认输出表存在
- [x] 5.4 运行 `openspec validate add-reused-weight-fusion-diagnostic-metrics --strict`
- [x] 5.5 运行 `openspec status --change add-reused-weight-fusion-diagnostic-metrics`，确认 apply 所需 artifacts 已完成
