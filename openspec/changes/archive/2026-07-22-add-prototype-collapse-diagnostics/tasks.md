## 1. Manifest 与中间层契约

- [x] 1.1 实现 checkpoint/config/fair-control 审计与 `checkpoint_manifest.yaml`，拒绝 outer-test/通信特权字段
- [x] 1.2 为 UMaskBeamJEPA 增加默认关闭的 `return_intermediates`，并记录真实 hook、alias、shape 到 layer manifest
- [x] 1.3 用 `conda run -n kd_mm_beam pytest tests/test_prototype_collapse_diagnostics.py -q` 验证默认 forward 兼容和 opt-in shape

## 2. 固定样本与特征缓存

- [x] 2.1 实现 3600 inner-train、900 inner-validation 的固定 sample/corruption manifest 与 checksum
- [x] 2.2 实现 float16 NPZ 分片抽取、断点跳过、clean/corrupt 配对、shape/dtype index 和禁用字段校验
- [x] 2.3 实现 `scripts/run_prototype_collapse_feature_extraction.sh`，记录实际 GPU、PID、日志且不含训练命令

## 3. 统计与 probe

- [x] 3.1 实现 D1 prototype geometry/usage/topology、D2 layer statistics/rank/drift 与图表
- [x] 3.2 实现 D3 beam/modality/corruption/severity/quality probes 与跨层 retention
- [x] 3.3 实现 D4 cross-modal CKA/retrieval 和 D5 single/LOMO/shuffle/replacement/gradient/unimodal evidence，并标记精确/近似/unavailable 状态
- [x] 3.4 实现 D6 R0/R1/R2 Router observability 与 sunny/rainy/foggy 分层
- [x] 3.5 实现 BC1-BC6 paired compression、centroid、residual probe、sensitivity、hidden score 和 virtual-gradient contraction

## 4. Runner、聚合与验证

- [x] 4.1 实现独立 D1-D6 薄入口、容错并行 runner、PID/日志和最终聚合器
- [x] 4.2 用 synthetic cache 的 `conda run -n kd_mm_beam pytest tests/test_prototype_collapse_diagnostics.py -q` 验证统计 schema、pairing 和判定
- [x] 4.3 运行 `openspec validate add-prototype-collapse-diagnostics --strict`、`make verify-quick` 和 `make verify-compile`

## 5. 本地完整诊断

- [x] 5.1 预处理固定输入并在 GPU0-5 抽取 A0/A1/B2/C0/C7 的完整 inner-train/inner-validation cache，不启动训练
- [x] 5.2 在受限线程 CPU 上运行 D1-D6 与 BC1-BC6，生成附件要求的 CSV、图表和分报告
- [x] 5.3 核对 1410 个分片 checksum 与输出完整性并生成唯一主方向的 `diagnostic_summary.md`，明确因果对照与 unavailable evidence
